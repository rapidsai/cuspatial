# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import NotImplementedPredicate
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
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


class CoversPredicateBase(EqualsPredicateBase):
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
    (Point, Point): CoversPredicateBase,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): PointLineStringIntersects,
    (Point, Polygon): CoversPredicateBase,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): NotImplementedPredicate,
    (LineString, Polygon): CoversPredicateBase,
    (Polygon, Point): CoversPredicateBase,
    (Polygon, MultiPoint): CoversPredicateBase,
    (Polygon, LineString): CoversPredicateBase,
    (Polygon, Polygon): CoversPredicateBase,
}
