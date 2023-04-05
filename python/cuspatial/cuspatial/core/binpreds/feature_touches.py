# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import NotImplementedPredicate
from cuspatial.core.binpreds.feature_contains import (
    ContainsPredicateBase,
    PointPointContains,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
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

    pass


class PointPointTouches(PointPointContains):
    """Points can't touch according to GeoPandas, so return False."""

    def _preprocess(self, lhs, rhs):
        return _false_series(len(lhs))


DispatchDict = {
    (Point, Point): PointPointTouches,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): NotImplementedPredicate,
    (Point, Polygon): TouchesPredicateBase,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): NotImplementedPredicate,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): NotImplementedPredicate,
    (LineString, Polygon): NotImplementedPredicate,
    (Polygon, Point): TouchesPredicateBase,
    (Polygon, MultiPoint): TouchesPredicateBase,
    (Polygon, LineString): TouchesPredicateBase,
    (Polygon, Polygon): TouchesPredicateBase,
}
