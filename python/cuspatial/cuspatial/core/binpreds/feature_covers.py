# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.core.binpreds.feature_intersects import (
    LineStringPointIntersects,
    PointLineStringIntersects,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class RootCovers(RootEquals):
    """Implements the covers predicate across different combinations of
    geometry types.  For example, a Point-Polygon covers predicate is
    defined in terms of a Point-Point equals predicate. The initial release
    implements covers predicates that depend only on the equals predicate, or
    depend on no predicate, such as impossible cases like
    `LineString.covers(Polygon)`.

    For this initial release, cover is supported for the following types:

    Point.covers(Point)
    Point.covers(Polygon)
    LineString.covers(Polygon)
    Polygon.covers(Point)
    Polygon.covers(MultiPoint)
    Polygon.covers(LineString)
    Polygon.covers(Polygon)
    """

    pass


DispatchDict = {
    (Point, Point): RootCovers,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): PointLineStringIntersects,
    (Point, Polygon): RootCovers,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): RootCovers,
    (Polygon, Point): RootCovers,
    (Polygon, MultiPoint): RootCovers,
    (Polygon, LineString): RootCovers,
    (Polygon, Polygon): RootCovers,
}
