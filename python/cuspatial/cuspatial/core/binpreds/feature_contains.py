# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import TypeVar

import cudf

from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_count,
    _basic_equals_all,
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
    _false_series,
    _linestrings_to_center_point,
    _open_polygon_rings,
    _pli_lines_to_multipoints,
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

    def _intersection_results_for_contains(self, lhs, rhs):
        pli = _basic_intersects_pli(lhs, rhs)
        pli_features = pli[1]
        if len(pli_features) == 0:
            return _zero_series(len(lhs))

        pli_offsets = cudf.Series(pli[0])

        # Convert the pli to multipoints for equality checking
        multipoints = _points_and_lines_to_multipoints(
            pli_features, pli_offsets
        )

        # A point in the rhs can be one of three possible states:
        # 1. It is in the interior of the lhs
        # 2. It is in the exterior of the lhs
        # 3. It is on the boundary of the lhs
        # This function tests if the point in the rhs is in the boundary
        # of the lhs
        intersect_equals_count = _basic_equals_count(rhs, multipoints)
        return intersect_equals_count

    def _compute_polygon_polygon_contains(self, lhs, rhs, preprocessor_result):
        lines_rhs = _open_polygon_rings(rhs)
        contains = _basic_contains_count(lhs, lines_rhs).reset_index(drop=True)
        intersects = self._intersection_results_for_contains(lhs, lines_rhs)
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

    def _test_interior(self, lhs, rhs):
        # We only need to test linestrings that are length 2.
        # Divide the linestring in half and test the point for containment
        # in the polygon.
        size_two = rhs.sizes == 2
        if (size_two).any():
            center_points = _linestrings_to_center_point(rhs[size_two])
            size_two_results = _false_series(len(lhs))
            size_two_results.iloc[rhs.index[size_two]] = (
                _basic_contains_count(lhs, center_points) > 0
            )
            return size_two_results
        else:
            return _false_series(len(lhs))

    def _compute_polygon_linestring_contains(
        self, lhs, rhs, preprocessor_result
    ):
        contains = _basic_contains_count(lhs, rhs).reset_index(drop=True)
        intersects = self._intersection_results_for_contains(lhs, rhs)

        # If a linestring has intersection but not containment, we need to
        # test if the linestring is in the interior of the polygon.
        final_result = _false_series(len(lhs))
        intersection_with_no_containment = (contains == 0) & (intersects != 0)
        interior_tests = self._test_interior(
            lhs[intersection_with_no_containment].reset_index(drop=True),
            rhs[intersection_with_no_containment].reset_index(drop=True),
        )
        interior_tests.index = intersection_with_no_containment[
            intersection_with_no_containment
        ].index
        # LineStrings that have intersection but no containment are set
        # according to the `intersection_with_no_containment` mask.
        final_result[intersection_with_no_containment] = interior_tests
        # LineStrings that do not are contained if the sum of intersecting
        # and containing points is greater than or equal to the number of
        # points that make up the linestring.
        final_result[~intersection_with_no_containment] = (
            contains + intersects >= rhs.sizes
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


class MultiPointMultiPointContains(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_equals_all(rhs, lhs)


class LineStringPointContains(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects(lhs, rhs)
        equals = _basic_equals_count(lhs, rhs) == rhs.sizes
        return intersects & ~equals


class LineStringLineStringContainsPredicate(BinPred):
    def _preprocess(self, lhs, rhs):
        # rhs_self_intersection is rhs where all colinear segments
        # are compacted into a single segment. This is necessary
        # because the intersection of lhs and rhs below will
        # compact colinear segments into a single segment, so
        # vertices in rhs will be lost and not countable.
        # Consider the following example:
        # lhs = [(0, 0), (1, 1)]
        # rhs = [(0, 0), (0.5, 0.5), (1, 1)]
        # The intersection of lhs and rhs is [(0, 0), (1, 1)].
        # rhs is contained by lhs if all of the vertices in the
        # intersection are in rhs. However, the intersection
        # does not contain the vertex (0.5, 0.5) in the original rhs.
        rhs_self_intersection = _basic_intersects_pli(rhs, rhs)
        rhs_no_segments = _points_and_lines_to_multipoints(
            rhs_self_intersection[1], rhs_self_intersection[0]
        )
        # Now intersect lhs and rhs and collect only the segments.
        pli = _basic_intersects_pli(lhs, rhs)
        lines = _pli_lines_to_multipoints(pli)
        # Every segment in B must be in the intersection segments.
        equals = (
            _basic_equals_count(lines, rhs_no_segments)
            == rhs_no_segments.sizes
        )
        return equals


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): PointPointContains,
    (Point, MultiPoint): MultiPointMultiPointContains,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): MultiPointMultiPointContains,
    (MultiPoint, MultiPoint): MultiPointMultiPointContains,
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
