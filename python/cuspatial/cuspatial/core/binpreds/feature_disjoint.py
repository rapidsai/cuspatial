# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_any,
    _basic_equals_any,
    _basic_intersects,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_intersects import IntersectsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class DisjointByWayOfContains(BinPred):
    def _preprocess(self, lhs, rhs):
        """Disjoint is the opposite of contains, so just implement contains
        and then negate the result.

        Used by:
        (Point, Polygon)
        (Polygon, Point)
        """
        return ~_basic_contains_any(lhs, rhs)


class PointPointDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        return ~_basic_equals_any(lhs, rhs)


class PointLineStringDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        """Disjoint is the opposite of intersects, so just implement intersects
        and then negate the result."""
        intersects = _basic_intersects(lhs, rhs)
        return ~intersects


class PointPolygonDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        return ~_basic_contains_any(lhs, rhs)


class LineStringPointDisjoint(PointLineStringDisjoint):
    def _preprocess(self, lhs, rhs):
        """Swap ordering for Intersects."""
        return super()._preprocess(rhs, lhs)


class LineStringLineStringDisjoint(IntersectsPredicateBase):
    def _postprocess(self, lhs, rhs, op_result):
        """Disjoint is the opposite of intersects, so just implement intersects
        and then negate the result."""
        result = super()._postprocess(lhs, rhs, op_result)
        return ~result


class LineStringPolygonDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        return ~_basic_contains_any(rhs, lhs)


class PolygonPolygonDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        return ~_basic_contains_any(lhs, rhs) & ~_basic_contains_any(rhs, lhs)


DispatchDict = {
    (Point, Point): PointPointDisjoint,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): PointLineStringDisjoint,
    (Point, Polygon): PointPolygonDisjoint,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): LineStringPolygonDisjoint,
    (LineString, Point): LineStringPointDisjoint,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringDisjoint,
    (LineString, Polygon): LineStringPolygonDisjoint,
    (Polygon, Point): DisjointByWayOfContains,
    (Polygon, MultiPoint): NotImplementedPredicate,
    (Polygon, LineString): DisjointByWayOfContains,
    (Polygon, Polygon): PolygonPolygonDisjoint,
}
