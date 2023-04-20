# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    PreprocessorResult,
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

    def _compute_predicate(
        self,
        lhs,
        rhs,
        preprocessor_result: PreprocessorResult,
    ):
        # contains = lhs._basic_contains_any(rhs)
        equals = lhs._basic_equals(rhs)
        intersects = lhs._basic_intersects(rhs)
        return equals | intersects


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
        # Intersection occurs
        intersects = lhs.intersects(rhs)
        # The linestring is contained but is not
        # contained properly, it crosses
        # This is the equivalent of crosses
        contains = rhs.contains(lhs)
        contains_properly = rhs.contains_properly(lhs)
        return intersects | (~contains & contains_properly)


class PolygonPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        # Intersection occurs
        intersects = lhs.intersects(rhs)
        # No points in the lhs are in the rhs
        contains = rhs._basic_contains_any(lhs)
        # Not equal
        equals_all = lhs._basic_equals_all(rhs)
        return intersects & ~contains & ~equals_all


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
    (Polygon, Point): TouchesPredicateBase,
    (Polygon, MultiPoint): TouchesPredicateBase,
    (Polygon, LineString): TouchesPredicateBase,
    (Polygon, Polygon): PolygonPolygonTouches,
}
