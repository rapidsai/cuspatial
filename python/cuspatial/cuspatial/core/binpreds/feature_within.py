# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class WithinPredicateBase(EqualsPredicateBase):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Within
    predicate is defined in terms of a Point-Point Contains predicate.
    Used by:
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    """

    pass


class WithinIntersectsPredicate(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = rhs._basic_intersects(lhs)
        equals = rhs._basic_equals(lhs)
        return intersects & ~equals


class PointLineStringWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = lhs.intersects(rhs)
        equals = lhs._basic_equals(rhs)
        return intersects & ~equals


class PointPolygonWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        return rhs.contains_properly(lhs)


class LineStringLineStringWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = rhs._basic_intersects(lhs)
        equals = rhs._basic_equals_all(lhs)
        return intersects & equals


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
