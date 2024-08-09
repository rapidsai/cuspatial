# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from typing import TypeVar

import cudf

from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_count,
    _basic_equals_any,
    _basic_equals_count,
    _basic_intersects,
    _basic_intersects_pli,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.contains_geometry_processor import (
    ContainsGeometryProcessor,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _open_polygon_rings,
    _pli_lines_to_multipoints,
    _pli_points_to_multipoints,
    _points_and_lines_to_multipoints,
    _zero_series,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_points,
    contains_only_polygons,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsPredicate(ContainsGeometryProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.allpairs = kwargs.get("allpairs", False)
        self.config.mode = kwargs.get("mode", "full")

    def _preprocess(self, lhs, rhs):
        preprocessor_result = super()._preprocess_multipoint_rhs(lhs, rhs)
        return self._compute_predicate(lhs, rhs, preprocessor_result)

    def _intersection_results_for_contains_linestring(self, lhs, rhs):
        pli = _basic_intersects_pli(lhs, rhs)

        # Convert the pli into points and multipoint intersections.
        multipoint_points = _pli_points_to_multipoints(pli)
        multipoint_lines = _pli_lines_to_multipoints(pli)

        # Count the point intersections that are equal to points in the
        # LineString
        # Count the linestring intersections that are equal to points in
        # the LineString
        return (
            _basic_equals_count(rhs, multipoint_points),
            _basic_equals_count(rhs, multipoint_lines),
        )

    def _intersection_results_for_contains_polygon(self, lhs, rhs):
        pli = _basic_intersects_pli(lhs, rhs)
        pli_features = pli[1]
        if len(pli_features) == 0:
            return _zero_series(len(lhs))

        pli_offsets = cudf.Series._from_column(pli[0])

        # Convert the pli to multipoints for equality checking
        multipoints = _points_and_lines_to_multipoints(
            pli_features, pli_offsets
        )

        intersect_equals_count = _basic_equals_count(rhs, multipoints)
        return intersect_equals_count

    def _compute_polygon_polygon_contains(self, lhs, rhs, preprocessor_result):
        lines_rhs = _open_polygon_rings(rhs)
        contains = _basic_contains_count(lhs, lines_rhs).reset_index(drop=True)
        intersects = self._intersection_results_for_contains_polygon(
            lhs, lines_rhs
        )
        # A closed polygon has an extra line segment that is not used in
        # counting the number of points. We need to subtract this from the
        # number of points in the polygon.
        multipolygon_part_offset = rhs.polygons.part_offset.take(
            rhs.polygons.geometry_offset
        )
        polygon_size_reduction = (
            multipolygon_part_offset[1:] - multipolygon_part_offset[:-1]
        )
        result = contains + intersects >= rhs.sizes - polygon_size_reduction
        return result

    def _compute_polygon_linestring_contains(
        self, lhs, rhs, preprocessor_result
    ):
        # Count the number of points in lhs that are properly contained by
        # rhs
        contains = _basic_contains_count(lhs, rhs).reset_index(drop=True)

        # Count the number of point intersections (line crossings) between
        # lhs and rhs.
        # Also count the number of perfectly overlapping linestring sections.
        # Each linestring overlap counts as two point overlaps.
        (
            point_intersects_count,
            linestring_intersects_count,
        ) = self._intersection_results_for_contains_linestring(lhs, rhs)

        # Subtract the length of the linestring intersections from the length
        # of the rhs linestring, then test that the sum of contained points
        # is equal to that adjusted rhs length.
        rhs_sizes_less_line_intersection_size = (
            rhs.sizes - linestring_intersects_count
        )
        rhs_sizes_less_line_intersection_size[
            rhs_sizes_less_line_intersection_size <= 0
        ] = 1
        final_result = contains + point_intersects_count == (
            rhs_sizes_less_line_intersection_size
        )

        return final_result

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        if contains_only_points(rhs):
            # Special case in GeoPandas, points are not contained
            # in the boundary of a polygon, so only return true if
            # the points are contained_properly.
            contains = _basic_contains_count(lhs, rhs).reset_index(drop=True)
            return contains > 0
        elif contains_only_linestrings(rhs):
            return self._compute_polygon_linestring_contains(
                lhs, rhs, preprocessor_result
            )
        elif contains_only_polygons(rhs):
            return self._compute_polygon_polygon_contains(
                lhs, rhs, preprocessor_result
            )
        else:
            raise NotImplementedError("Invalid rhs for contains operation")


class PointPointContains(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_equals_any(lhs, rhs)


class LineStringPointContains(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects(lhs, rhs)
        equals = _basic_equals_any(lhs, rhs)
        return intersects & ~equals


class LineStringLineStringContainsPredicate(BinPred):
    def _preprocess(self, lhs, rhs):
        pli = _basic_intersects_pli(lhs, rhs)
        points = _points_and_lines_to_multipoints(pli[1], pli[0])
        # Every point in B must be in the intersection
        equals = _basic_equals_count(rhs, points) == rhs.sizes
        return equals


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): PointPointContains,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointContains,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringContainsPredicate,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsPredicate,
    (Polygon, MultiPoint): ContainsPredicate,
    (Polygon, LineString): ContainsPredicate,
    (Polygon, Polygon): ContainsPredicate,
}
