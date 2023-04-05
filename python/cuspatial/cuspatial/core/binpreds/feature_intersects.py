# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import NotImplementedPredicate
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class IntersectsPredicateBase(EqualsPredicateBase):
    """Base class for binary predicates that are defined in terms of
    the intersects basic predicate. These predicates are defined in terms
    of the equals basic predicate. The type dispatches here that depend
    on `IntersectsPredicateBase` use the `PredicateEquals` class for their
    complete implementation, unmodified.

    point.intersects(polygon) is equivalent to polygon.contains(point)
    with the left and right hand sides reversed.
    """

    pass


class PointPolygonIntersects(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Swap LHS and RHS and call the normal contains processing."""
        self.lhs = rhs
        self.rhs = lhs
        return super()._preprocess(rhs, lhs)


""" Type dispatch dictionary for intersects binary predicates. """
DispatchDict = {
    (Point, Point): IntersectsPredicateBase,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): NotImplementedPredicate,
    (Point, Polygon): PointPolygonIntersects,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): NotImplementedPredicate,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): NotImplementedPredicate,
    (LineString, Polygon): NotImplementedPredicate,
    (Polygon, Point): IntersectsPredicateBase,
    (Polygon, MultiPoint): IntersectsPredicateBase,
    (Polygon, LineString): IntersectsPredicateBase,
    (Polygon, Polygon): IntersectsPredicateBase,
}
