# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.basic_predicates import (
    _basic_equals_all,
    _basic_equals_any,
    _basic_intersects,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class WithinPredicateBase(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_equals_all(lhs, rhs)


class WithinIntersectsPredicate(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects(rhs, lhs)
        equals = _basic_equals_any(rhs, lhs)
        return intersects & ~equals


class PointLineStringWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = lhs.intersects(rhs)
        equals = _basic_equals_any(lhs, rhs)
        return intersects & ~equals


class PointPolygonWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        return rhs.contains_properly(lhs)


class LineStringLineStringWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        contains = rhs.contains(lhs)
        return contains


class LineStringPolygonWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        return rhs.contains(lhs)


class PolygonPolygonWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        return rhs.contains(lhs)


DispatchDict = {
    (Point, Point): WithinPredicateBase,
    (Point, MultiPoint): WithinIntersectsPredicate,
    (Point, LineString): PointLineStringWithin,
    (Point, Polygon): PointPolygonWithin,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): WithinIntersectsPredicate,
    (MultiPoint, Polygon): PolygonPolygonWithin,
    (LineString, Point): ImpossiblePredicate,
    (LineString, MultiPoint): WithinIntersectsPredicate,
    (LineString, LineString): LineStringLineStringWithin,
    (LineString, Polygon): LineStringPolygonWithin,
    (Polygon, Point): WithinPredicateBase,
    (Polygon, MultiPoint): WithinPredicateBase,
    (Polygon, LineString): WithinPredicateBase,
    (Polygon, Polygon): PolygonPolygonWithin,
}
