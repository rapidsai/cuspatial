# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_count,
    _basic_contains_properly_any,
    _basic_equals,
    _basic_equals_all,
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
    """Base class for binary predicates that use the contains predicate
    to implement the touches predicate. For example, a Point-Polygon
    Touches predicate is defined in terms of a Point-Polygon Contains
    predicate.

    Used by:
    (Point, Polygon)
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    (Polygon, Polygon)
    """

    def _preprocess(self, lhs, rhs):
        equals = _basic_equals(lhs, rhs)
        return equals


class PointLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_equals(lhs, rhs)


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
        # Point is equal
        equals = _basic_equals(lhs, rhs)
        # Linestrings are not equal
        equals_all = _basic_equals_all(lhs, rhs)
        # Linestrings do not cross
        crosses = ~lhs.crosses(rhs)
        return equals & crosses & ~equals_all


class LineStringPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        pli = _basic_intersects_pli(lhs, rhs)
        if len(pli[1]) == 0:
            return _false_series(len(lhs))
        intersections = _points_and_lines_to_multipoints(pli[1], pli[0])
        # A touch can only occur if the point in the intersection
        # is equal to a point in the linestring, it must
        # terminate in the boundary of the polygon.
        equals = _basic_equals_count(intersections, lhs) > 0
        intersects = _basic_intersects_count(lhs, rhs)
        contains = rhs.contains(lhs)
        contains_any = _basic_contains_properly_any(rhs, lhs)
        intersects = (intersects == 1) | (intersects == 2)
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
        intersects = _basic_intersects_count(lhs, rhs) == 1
        return contains_lhs_none & contains_rhs_none & intersects


DispatchDict = {
    (Point, Point): ImpossiblePredicate,
    (Point, MultiPoint): TouchesPredicateBase,
    (Point, LineString): PointLineStringTouches,
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
