# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import TypeVar

import cudf

from cuspatial.core.binpreds.binpred_interface import (
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.complex_geometry_predicate import (
    ComplexGeometryPredicate,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
    _linestrings_to_center_point,
    _open_polygon_rings,
    _points_and_lines_to_multipoints,
    _zero_series,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_points,
    contains_only_polygons,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsPredicateBase(ComplexGeometryPredicate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.allpairs = kwargs.get("allpairs", False)
        self.config.mode = kwargs.get("mode", "full")

    def _preprocess(self, lhs, rhs):
        preprocessor_result = super()._preprocess_multi(lhs, rhs)
        return self._compute_predicate(lhs, rhs, preprocessor_result)

    def _intersection_results_for_contains(self, lhs, rhs):
        pli = lhs._basic_intersects_pli(rhs)
        pli_features = pli[1]
        if len(pli_features) == 0:
            return _zero_series(len(lhs))

        pli_offsets = cudf.Series(pli[0])

        multipoints = _points_and_lines_to_multipoints(
            pli_features, pli_offsets
        )

        intersect_equals_count = rhs._basic_equals_count(multipoints)
        breakpoint()
        return intersect_equals_count

    def _compute_polygon_polygon_contains(self, lhs, rhs, preprocessor_result):
        lines_rhs = _open_polygon_rings(rhs)
        contains = lhs._basic_contains_count(lines_rhs).reset_index(drop=True)
        intersects = self._intersection_results_for_contains(lhs, lines_rhs)
        polygon_size_reduction = rhs.polygons.part_offset.take(
            rhs.polygons.geometry_offset[1:]
        ) - rhs.polygons.part_offset.take(rhs.polygons.geometry_offset[:-1])
        breakpoint()
        return contains + intersects >= rhs.sizes - polygon_size_reduction

    def _compute_polygon_linestring_contains(
        self, lhs, rhs, preprocessor_result
    ):
        contains = lhs._basic_contains_count(rhs).reset_index(drop=True)
        intersects = self._intersection_results_for_contains(lhs, rhs)
        if (contains == 0).all() and (intersects != 0).all():
            # The hardest case. We need to check if the linestring is
            # contained in the boundary of the polygon, the interior,
            # or the exterior.
            # We only need to test linestrings that are length 2.
            # Divide the linestring in half and test the point for containment
            # in the polygon.

            if (rhs.sizes == 2).any():
                center_points = _linestrings_to_center_point(
                    rhs[rhs.sizes == 2]
                )
                size_two_results = _false_series(len(lhs))
                size_two_results[rhs.sizes == 2] = (
                    lhs._basic_contains_count(center_points) > 0
                )
                return size_two_results
            else:
                line_intersections = _false_series(len(lhs))
                line_intersections[intersects == rhs.sizes] = True
                return line_intersections
        return contains + intersects >= rhs.sizes

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        if contains_only_points(rhs):
            # Special case in GeoPandas, points are not contained
            # in the boundary of a polygon.
            contains = lhs._basic_contains_count(rhs).reset_index(drop=True)
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


class ContainsPredicate(ContainsPredicateBase):
    def _compute_results(self, lhs, rhs, preprocessor_result):
        return lhs._contains(rhs)


class PointPointContains(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        return lhs._basic_equals(rhs)


class LineStringMultiPointContainsPredicate(ContainsPredicateBase):
    def _compute_results(self, lhs, rhs, preprocessor_result):
        return lhs._linestring_multipoint_contains(rhs)


class LineStringLineStringContainsPredicate(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        count = lhs._basic_equals_count(rhs)
        return count == rhs.sizes


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
    (LineString, Point): ContainsPredicateBase,
    (LineString, MultiPoint): LineStringMultiPointContainsPredicate,
    (LineString, LineString): LineStringLineStringContainsPredicate,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsPredicateBase,
    (Polygon, MultiPoint): ContainsPredicateBase,
    (Polygon, LineString): ContainsPredicateBase,
    (Polygon, Polygon): ContainsPredicateBase,
}
