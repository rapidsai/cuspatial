# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.basic_predicates import (
    _basic_equals,
    _basic_intersects,
    _basic_intersects_count,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import IntersectsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class CrossesPredicateBase(EqualsPredicateBase):
    """Base class for binary predicates that are defined in terms of a
    the equals binary predicate. For example, a Point-Point Crosses
    predicate is defined in terms of a Point-Point Equals predicate.

    Used by:
    (Point, Polygon)
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    (Polygon, Polygon)
    """

    pass


class CrossesByIntersectionPredicate(IntersectsPredicateBase):
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        intersects = _basic_intersects(rhs, lhs)
        equals = _basic_equals(rhs, lhs)
        return intersects & ~equals


class LineStringPolygonCrosses(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects_count(rhs, lhs) > 1
        touches = rhs.touches(lhs)
        contains = rhs.contains(lhs)
        return ~touches & intersects & ~contains


class PolygonLineStringCrosses(LineStringPolygonCrosses):
    def _preprocess(self, lhs, rhs):
        return super()._preprocess(rhs, lhs)


class PointPointCrosses(CrossesPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Points can't cross other points, so we return False."""
        return _false_series(len(lhs))


DispatchDict = {
    (Point, Point): PointPointCrosses,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): CrossesPredicateBase,
    (MultiPoint, Point): ImpossiblePredicate,
    (MultiPoint, MultiPoint): ImpossiblePredicate,
    (MultiPoint, LineString): ImpossiblePredicate,
    (MultiPoint, Polygon): ImpossiblePredicate,
    (LineString, Point): ImpossiblePredicate,
    (LineString, MultiPoint): ImpossiblePredicate,
    (LineString, LineString): CrossesByIntersectionPredicate,
    (LineString, Polygon): LineStringPolygonCrosses,
    (Polygon, Point): CrossesPredicateBase,
    (Polygon, MultiPoint): CrossesPredicateBase,
    (Polygon, LineString): PolygonLineStringCrosses,
    (Polygon, Polygon): ImpossiblePredicate,
}
