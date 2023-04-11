# Copyright (c) 2023, NVIDIA CORPORATION.


import cupy as cp

import cudf

import cuspatial
from cuspatial.core.binops.intersection import pairwise_linestring_intersection
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    IntersectsOpResult,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class IntersectsPredicateBase(BinPred):
    """Base class for binary predicates that are defined in terms of
    the intersection primitive.
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

    def _get_intersecting_geometry_indices(self, lhs, op_result):
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
        match_indices = self._get_intersecting_geometry_indices(lhs, op_result)
        result = _false_series(len(lhs))
        if len(op_result.result[1]) > 0:
            result[match_indices] = True
        return result


class IntersectsByEquals(EqualsPredicateBase):
    pass


class PointPolygonIntersects(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Swap LHS and RHS and call the normal contains processing."""
        self.lhs = rhs
        self.rhs = lhs
        return super()._preprocess(rhs, lhs)


class LineStringPointIntersects(IntersectsPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings by making a linestring that has
        the same start and end point."""
        x = cp.repeat(rhs.points.x, 2)
        y = cp.repeat(rhs.points.y, 2)
        xy = cudf.DataFrame({"x": x, "y": y}).interleave_columns()
        parts = cp.arange((len(lhs) + 1)) * 2
        geometries = cp.arange(len(lhs) + 1)
        ls_rhs = cuspatial.GeoSeries.from_linestrings_xy(xy, parts, geometries)
        return self._compute_predicate(
            lhs, ls_rhs, PreprocessorResult(lhs, ls_rhs)
        )


class LineStringMultiPointIntersects(IntersectsPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Convert rhs to linestrings."""
        xy = rhs.multipoints.xy
        parts = rhs.multipoints.geometry_offset
        geometries = cp.arange(len(lhs) + 1)
        ls_rhs = cuspatial.GeoSeries.from_linestrings_xy(xy, parts, geometries)
        return self._compute_predicate(
            lhs, ls_rhs, PreprocessorResult(lhs, ls_rhs)
        )


class PointLineStringIntersects(LineStringPointIntersects):
    def _preprocess(self, lhs, rhs):
        """Swap LHS and RHS and call the normal contains processing."""
        return super()._preprocess(rhs, lhs)


class LineStringPointIntersects(IntersectsPredicateBase):
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


""" Type dispatch dictionary for intersects binary predicates. """
DispatchDict = {
    (Point, Point): IntersectsByEquals,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): PointLineStringIntersects,
    (Point, Polygon): PointPolygonIntersects,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): LineStringMultiPointIntersects,
    (LineString, LineString): IntersectsPredicateBase,
    (LineString, Polygon): NotImplementedPredicate,
    (Polygon, Point): NotImplementedPredicate,
    (Polygon, MultiPoint): NotImplementedPredicate,
    (Polygon, LineString): NotImplementedPredicate,
    (Polygon, Polygon): NotImplementedPredicate,
}
