# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import cupy as cp

import cudf

from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_count,
    _basic_contains_properly_any,
    _basic_equals_all,
    _basic_equals_any,
    _basic_equals_count,
    _basic_intersects,
    _basic_intersects_count,
    _basic_intersects_pli,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
)
from cuspatial.core.binpreds.feature_contains import ContainsPredicate
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
    _pli_points_to_multipoints,
    _points_and_lines_to_multipoints,
)


class TouchesPredicateBase(ContainsPredicate):
    """
    If any point is shared between the following geometry types, they touch:

    Used by:
    (Point, MultiPoint)
    (Point, LineString)
    (MultiPoint, Point)
    (MultiPoint, MultiPoint)
    (MultiPoint, LineString)
    (MultiPoint, Polygon)
    (LineString, Point)
    (LineString, MultiPoint)
    (Polygon, MultiPoint)
    """

    def _preprocess(self, lhs, rhs):
        return _basic_equals_any(lhs, rhs)


class PointPolygonTouches(ContainsPredicate):
    def _preprocess(self, lhs, rhs):
        # Reverse argument order.
        equals_all = _basic_equals_all(rhs, lhs)
        touches = _basic_intersects(rhs, lhs)
        return ~equals_all & touches


class LineStringLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        """A and B have at least one point in common, and the common points
        lie in at least one boundary"""

        # First compute pli which will contain points for line crossings and
        # linestrings for overlapping segments.
        pli = _basic_intersects_pli(lhs, rhs)
        offsets = cudf.Series._from_column(pli[0])
        pli_geometry_count = offsets[1:].reset_index(drop=True) - offsets[
            :-1
        ].reset_index(drop=True)
        indices = (
            cudf.Series(cp.arange(len(pli_geometry_count)))
            .repeat(pli_geometry_count)
            .reset_index(drop=True)
        )

        # In order to be a touch, all of the intersecting geometries
        # for a particular row must be points.
        pli_types = pli[1]._column._meta.input_types
        point_intersection = _false_series(len(lhs))
        only_points_in_intersection = (
            pli_types.groupby(indices).sum().sort_index() == 0
        )
        point_intersection.iloc[
            only_points_in_intersection.index
        ] = only_points_in_intersection

        # Finally, we need to check if the points in the intersection
        # are equal to endpoints of either linestring.
        points = _points_and_lines_to_multipoints(pli[1], pli[0])
        equals_lhs = _basic_equals_count(points, lhs) > 0
        equals_rhs = _basic_equals_count(points, rhs) > 0
        touches = point_intersection & (equals_lhs | equals_rhs)
        return touches & ~lhs.crosses(rhs)


class LineStringPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects_count(lhs, rhs) > 0
        contains = rhs.contains(lhs)
        contains_any = _basic_contains_properly_any(rhs, lhs)

        pli = _basic_intersects_pli(lhs, rhs)
        if len(pli[1]) == 0:
            return _false_series(len(lhs))
        points = _pli_points_to_multipoints(pli)
        # A touch can only occur if the point in the intersection
        # is equal to a point in the linestring: it must
        # terminate in the boundary of the polygon.
        equals = _basic_equals_count(points, lhs) == points.sizes

        return equals & intersects & ~contains & ~contains_any


class PolygonPointTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects(lhs, rhs)
        return intersects


class PolygonLineStringTouches(LineStringPolygonTouches):
    def _preprocess(self, lhs, rhs):
        return super()._preprocess(rhs, lhs)


class PolygonPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        contains_lhs_none = _basic_contains_count(lhs, rhs) == 0
        contains_rhs_none = _basic_contains_count(rhs, lhs) == 0
        contains_lhs = lhs.contains(rhs)
        contains_rhs = rhs.contains(lhs)
        equals = lhs.geom_equals(rhs)
        intersect_count = _basic_intersects_count(lhs, rhs)
        intersects = (intersect_count > 0) & (intersect_count < rhs.sizes - 1)
        result = (
            ~equals
            & contains_lhs_none
            & contains_rhs_none
            & ~contains_lhs
            & ~contains_rhs
            & intersects
        )
        return result


DispatchDict = {
    (Point, Point): ImpossiblePredicate,
    (Point, MultiPoint): TouchesPredicateBase,
    (Point, LineString): TouchesPredicateBase,
    (Point, Polygon): PointPolygonTouches,
    (MultiPoint, Point): TouchesPredicateBase,
    (MultiPoint, MultiPoint): TouchesPredicateBase,
    (MultiPoint, LineString): TouchesPredicateBase,
    (MultiPoint, Polygon): TouchesPredicateBase,
    (LineString, Point): TouchesPredicateBase,
    (LineString, MultiPoint): TouchesPredicateBase,
    (LineString, LineString): LineStringLineStringTouches,
    (LineString, Polygon): LineStringPolygonTouches,
    (Polygon, Point): PolygonPointTouches,
    (Polygon, MultiPoint): TouchesPredicateBase,
    (Polygon, LineString): PolygonLineStringTouches,
    (Polygon, Polygon): PolygonPolygonTouches,
}
