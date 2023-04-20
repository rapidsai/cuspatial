# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import IntersectsPredicateBase
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


class WithinIntersectsPredicate(IntersectsPredicateBase):
    def _preprocess(self, lhs, rhs):
        intersects = rhs._basic_intersects(lhs)
        equals = rhs._basic_equals(lhs)
        return intersects & ~equals


class PointLineStringWithin(WithinIntersectsPredicate):
    def _preprocess(self, lhs, rhs):
        # Note the order of arguments is reversed.
        return super()._preprocess(rhs, lhs)


class PointPolygonWithin(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        return rhs.contains_properly(lhs)


class LineStringLineStringWithin(IntersectsPredicateBase):
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        intersects = rhs._basic_intersects(lhs)
        equals = rhs._basic_equals_all(lhs)
        return intersects & equals


class LineStringPolygonWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        return rhs.contains(rhs)


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
    (LineString, Point): WithinIntersectsPredicate,
    (LineString, MultiPoint): WithinIntersectsPredicate,
    (LineString, LineString): LineStringLineStringWithin,
    (LineString, Polygon): LineStringPolygonWithin,
    (Polygon, Point): WithinPredicateBase,
    (Polygon, MultiPoint): WithinPredicateBase,
    (Polygon, LineString): WithinPredicateBase,
    (Polygon, Polygon): PolygonPolygonWithin,
}
