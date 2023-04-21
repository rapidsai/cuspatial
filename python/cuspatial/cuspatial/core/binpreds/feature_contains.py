# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import TypeVar

import cupy as cp
import numpy as np

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
        pli_offsets = cudf.Series(pli[0])
        # Which feature goes with which offset?
        pli_sizes = pli_offsets[1:].reset_index(drop=True) - pli_offsets[
            :-1
        ].reset_index(drop=True)
        # Have to use host to create the offsets mapping
        pli_mapping = cp.array(
            np.arange(len(lhs)).repeat(pli_sizes.values_host)
        )

        # This mapping identifies which intersect feature belongs to which
        # intersection.

        points_mask = pli_features.type == "Point"
        lines_mask = pli_features.type == "Linestring"

        points = pli_features[points_mask]
        lines = pli_features[lines_mask]

        final_intersection_count = _zero_series(len(lhs))
        from cuspatial.core.geoseries import GeoSeries

        #  Write a new method, _points_and_lines_to_multipoints that condenses
        # The result into a single multipoint that can be worked with.
        multipoints = _points_and_lines_to_multipoints(pli_features)

        if len(lines) > 0:
            # This is wrong. If a linestring is in a single intersection,
            # it will tile out to all of the features. It needs to be
            # compared only against the matching feature. pli_mapping
            # determines which features match which intersections.

            multipoints = GeoSeries.from_multipoints_xy(
                lines.lines.xy, pli_offsets * 2
            )
            lines_intersect_equals_count = multipoints._basic_equals_count(rhs)
            final_intersection_count.iloc[
                pli_mapping[lines_mask]
            ] = lines_intersect_equals_count[pli_mapping[lines_mask]]
            breakpoint()
        if len(points) > 0:
            # Each point falls on the edge of the polygon and is in the
            # boundary.
            multipoints = GeoSeries.from_multipoints_xy(
                points.points.xy.tile(len(lhs)),
                cp.arange(len(lhs) + 1) * len(points),
            )
            points_intersect_equals_count = multipoints._basic_equals_count(
                rhs
            ) // len(lhs)
            final_intersection_count.iloc[
                pli_mapping[points_mask]
            ] = points_intersect_equals_count[pli_mapping[points_mask]]
            breakpoint()
        # TODO Have to use .iloc here because of a bug in cudf
        return final_intersection_count

    def _compute_polygon_polygon_contains(self, lhs, rhs, preprocessor_result):
        lines_rhs = _open_polygon_rings(rhs)
        contains = lhs._basic_contains_count(lines_rhs).reset_index(drop=True)
        intersects = self._intersection_results_for_contains(lhs, lines_rhs)
        polygon_size_reduction = 1
        breakpoint()
        return contains + intersects >= rhs.sizes - polygon_size_reduction

    def _compute_polygon_linestring_contains(
        self, lhs, rhs, preprocessor_result
    ):
        contains = lhs._basic_contains_count(rhs).reset_index(drop=True)
        if (contains == 0).all():
            # If a linestring only intersects with the boundary of a polygon,
            # it is not contained.
            return rhs.sizes == 2
        intersects = self._intersection_results_for_contains(lhs, rhs)
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
