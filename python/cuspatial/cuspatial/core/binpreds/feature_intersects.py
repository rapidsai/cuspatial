# Copyright (c) 2023, NVIDIA CORPORATION.


import cupy as cp
from shapely.geometry import LineString as ShapelyLineString

import cudf

import cuspatial
from cuspatial.core.binops.intersection import pairwise_linestring_intersection
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    IntersectsOpResult,
    NotImplementedRoot,
    PreprocessorResult,
)
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class RootIntersects(BinPred):
    """Base class for binary predicates that are defined in terms of
    the intersects basic predicate.
    """

    def _preprocess(self, lhs, rhs):
        """Convert input lhs and rhs into LineStrings.

        The intersection basic predicate requires that the input
        geometries be LineStrings or MultiLineStrings. This function
        converts the input geometries into LineStrings and stores them
        in the PreprocessorResult object that is passed into
        _compute_predicate.

        Parameters
        ----------
        lhs : GeoSeries
            The left hand side of the binary predicate.
        rhs : GeoSeries
            The right hand side of the binary predicate.

        Returns
        -------
        GeoSeries
            A boolean Series containing the computed result of the
            final binary predicate operation,
        """
        return self._compute_predicate(lhs, rhs, PreprocessorResult(lhs, rhs))

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        """Compute the predicate using the intersects basic predicate.
        lhs and rhs must both be LineStrings or MultiLineStrings.
        """
        basic_result = pairwise_linestring_intersection(
            preprocessor_result.lhs, preprocessor_result.rhs
        )
        computed_result = IntersectsOpResult(basic_result)
        return self._postprocess(lhs, rhs, computed_result)

    def _get_match_indices(self, lhs, op_result):
        """Naively computes the indices of matches by constructing
        a set of lengths from the returned offsets buffer, then
        returns an integer index for all of the offset sizes that
        are larger than 0."""
        is_offsets = cudf.Series(op_result.result[0])
        is_sizes = is_offsets[1:].reset_index(drop=True) - is_offsets[
            :-1
        ].reset_index(drop=True)
        return cp.arange(len(lhs))[is_sizes > 0]

    def _linestrings_from_polygons(self, geoseries):
        xy = geoseries.polygons.xy
        parts = geoseries.polygons.part_offset.take(
            geoseries.polygons.geometry_offset
        )
        rings = geoseries.polygons.ring_offset
        return cuspatial.GeoSeries.from_linestrings_xy(
            xy,
            rings,
            parts,
        )

    def _postprocess(self, lhs, rhs, op_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        match_indices = self._get_match_indices(lhs, op_result)
        result = _false_series(lhs)
        if len(op_result.result[1]) > 0 and len(lhs) == 1:
            result[0] = True
        elif len(op_result.result[1]) > 0:
            result[match_indices] = True
        return result


class IntersectsByEquals(RootEquals):
    pass


class PointPolygonIntersects(RootContains):
    def _preprocess(self, lhs, rhs):
        """Swap LHS and RHS and call the normal contains processing."""
        self.lhs = rhs
        self.rhs = lhs
        return super()._preprocess(rhs, lhs)


class LineStringPointIntersects(RootIntersects):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings."""
        x = cp.repeat(rhs.points.x, 2)
        y = cp.repeat(rhs.points.y, 2)
        xy = cudf.DataFrame({"x": x, "y": y}).interleave_columns()
        parts = cp.arange((len(lhs) + 1)) * 2
        geometries = cp.arange(len(lhs) + 1)
        ls_rhs = cuspatial.GeoSeries.from_linestrings_xy(xy, parts, geometries)
        return self._compute_predicate(
            lhs, ls_rhs, PreprocessorResult(lhs, ls_rhs)
        )


class LineStringMultiPointIntersects(RootIntersects):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings."""
        xy = rhs.multipoints.xy
        parts = rhs.multipoints.geometry_offset
        geometries = cp.arange(len(lhs) + 1)
        ls_rhs = cuspatial.GeoSeries.from_linestrings_xy(xy, parts, geometries)
        return self._compute_predicate(
            lhs, ls_rhs, PreprocessorResult(lhs, ls_rhs)
        )

    def _postprocess(self, lhs, rhs, op_result):
        """When a multipoint intersects with a linestring, all of the
        points in the intersection need to be tested for equality with
        the linestring.

        This code uses the result of the intersection to determine
        which points from the rhs to test for membership in the lhs.

        Every point in the intersection is tested for membership in
        the lhs. If the point is not in the lhs, the result is set to
        False. If the point is in the lhs, the result is set to True.

        This iterates on the number of results in the intersection,
        which is not ideal. This code calls for the existence of a
        libcuspatial function that can return the points in rhs
        that are equal to points in the lhs.
        """
        match_indices = self._get_match_indices(lhs, op_result)
        result = super()._postprocess(lhs, rhs, op_result)
        intersections = op_result.result[1]
        x_coords = rhs.lines.x
        y_coords = rhs.lines.y
        result = _false_series(lhs)
        for idx in range(len(intersections)):
            if isinstance(intersections[idx], ShapelyLineString):
                result[match_indices[idx]] = True
                continue
            x_match = intersections.points.x[idx] == x_coords
            if x_match.sum() > 0:
                y_match = intersections.points.y[idx] == y_coords
                if (y_match & x_match).sum() > 0:
                    result[match_indices[idx]] = True
        return result


class LineStringPolygonIntersects(RootIntersects):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings."""
        ls_rhs = self._linestrings_from_polygons(rhs)
        return self._compute_predicate(
            lhs, ls_rhs, PreprocessorResult(lhs, ls_rhs)
        )


class PolygonLineStringIntersects(RootIntersects):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings."""
        ls_lhs = self._linestrings_from_polygons(lhs)
        return self._compute_predicate(
            ls_lhs, rhs, PreprocessorResult(ls_lhs, rhs)
        )


class PolygonPolygonIntersects(RootIntersects):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings."""
        ls_lhs = self._linestrings_from_polygons(lhs)
        ls_rhs = self._linestrings_from_polygons(rhs)
        return self._compute_predicate(
            ls_lhs, ls_rhs, PreprocessorResult(ls_lhs, ls_rhs)
        )


""" Type dispatch dictionary for intersects binary predicates. """
DispatchDict = {
    (Point, Point): IntersectsByEquals,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): PointPolygonIntersects,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): LineStringMultiPointIntersects,
    (LineString, LineString): RootIntersects,
    (LineString, Polygon): LineStringPolygonIntersects,
    (Polygon, Point): IntersectsByEquals,
    (Polygon, MultiPoint): IntersectsByEquals,
    (Polygon, LineString): PolygonLineStringIntersects,
    (Polygon, Polygon): PolygonPolygonIntersects,
}
