# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
)
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class TouchesPredicateBase(ContainsPredicateBase):
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
        equals = lhs._basic_equals(rhs)
        return equals


class PointLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        return lhs._basic_equals(rhs)


class PointPolygonTouches(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        # Reverse argument order.
        equals_all = rhs._basic_equals_all(lhs)
        touches = rhs._basic_intersects(lhs)
        return ~equals_all & touches


class LineStringLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        """A and B have at least one point in common, and the common points
        lie in at least one boundary"""
        # Point is equal
        equals = lhs._basic_equals(rhs)
        # Linestrings are not equal
        equals_all = lhs._basic_equals_all(rhs)
        # Linestrings do not cross
        crosses = ~lhs.crosses(rhs)
        return equals & crosses & ~equals_all


class LineStringPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = lhs._basic_intersects_count(rhs) == 1
        contains_none = ~lhs.contains_properly(rhs)
        return intersects & contains_none


class PolygonPointTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = lhs._basic_intersects(rhs)
        return intersects


class PolygonLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        # Intersection occurs
        intersects = lhs._basic_intersects_count(rhs) == 1
        contains_none = ~lhs.contains_properly(rhs)
        return intersects & contains_none


class PolygonPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        contains_lhs_none = lhs._basic_contains_count(rhs) == 0
        contains_rhs_none = rhs._basic_contains_count(lhs) == 0
        intersects = lhs._basic_intersects_count(rhs) == 1
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
