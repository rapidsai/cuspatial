# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
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
    """

    pass


class PointPointCrosses(CrossesPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Points can't cross other points, so we return False."""
        return _false_series(len(lhs))


DispatchDict = {
    (Point, Point): PointPointCrosses,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): NotImplementedPredicate,
    (Point, Polygon): CrossesPredicateBase,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): ImpossiblePredicate,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): NotImplementedPredicate,
    (LineString, Polygon): NotImplementedPredicate,
    (Polygon, Point): CrossesPredicateBase,
    (Polygon, MultiPoint): CrossesPredicateBase,
    (Polygon, LineString): CrossesPredicateBase,
    (Polygon, Polygon): CrossesPredicateBase,
}
