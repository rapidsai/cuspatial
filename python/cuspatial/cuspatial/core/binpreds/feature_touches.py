# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import (
    PointPointContains,
    RootContains,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class RootTouches(RootContains):
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
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootTouches,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): RootTouches,
    (Polygon, MultiPoint): RootTouches,
    (Polygon, LineString): RootTouches,
    (Polygon, Polygon): RootTouches,
}
