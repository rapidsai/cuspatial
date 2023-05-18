# Copyright (c) 2023, NVIDIA CORPORATION.

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
        equals = _basic_equals_any(lhs, rhs)
        return equals


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
        # Linestrings are not equal
        equals_all = _basic_equals_all(lhs, rhs)
        # Linestrings do not cross
        crosses = lhs.crosses(rhs)
        intersects = lhs.intersects(rhs)
        return intersects & ~crosses & ~equals_all


class LineStringPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        pli = _basic_intersects_pli(lhs, rhs)
        if len(pli[1]) == 0:
            return _false_series(len(lhs))
        intersections = _points_and_lines_to_multipoints(pli[1], pli[0])
        # A touch can only occur if the point in the intersection
        # is equal to a point in the linestring: it must
        # terminate in the boundary of the polygon.
        equals = _basic_equals_count(intersections, lhs) > 0
        intersects = _basic_intersects_count(lhs, rhs)
        intersects = (intersects == 1) | (intersects == 2)
        contains = rhs.contains(lhs)
        contains_any = _basic_contains_properly_any(rhs, lhs)
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
        equals = lhs.geom_equals(rhs)
        intersects = _basic_intersects_count(lhs, rhs) > 0
        return ~equals & contains_lhs_none & contains_rhs_none & intersects


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
