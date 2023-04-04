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
    """
    A point touches a point if they are equal.
    A point touches a polygon if it is contained by the polygon.
    A point touches a linestring if it is contained by the linestring.
    A point touches a multipoint if it is contained by the multipoint.
    A multipoint touches a point if it contains the point.
    A multipoint touches a multipoint if they have at least one point in
    common.
    A multipoint touches a linestring if it contains the linestring.
    A multipoint touches a polygon if it contains the polygon.
    A linestring touches a point if it contains the point.
    A linestring touches a multipoint if it is contained by the multipoint.
    A linestring touches a linestring if they have at least one point in
        common.
    A linestring touches a polygon if it contains the polygon.
    A polygon touches a point if it contains the point.
    A polygon touches a multipoint if it is contained by the multipoint.
    A polygon touches a linestring if it is contained by the linestring.
    A polygon touches a polygon if they have at least one point in common."""

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
