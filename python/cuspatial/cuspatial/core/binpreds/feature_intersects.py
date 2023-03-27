# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class RootIntersects(RootEquals):
    """Base class for binary predicates that are defined in terms of
    the intersects basic predicate. These predicates are defined in terms
    of the equals basic predicate. The type dispatches here that depend
    on `RootIntersects` use the `RootEquals` class for their complete
    implementation, unmodified.

    point.intersects(polygon) is equivalent to polygon.contains(point)
    with the left and right hand sides reversed.
    """

    pass


class PointPolygonIntersects(RootContains):
    def _preprocess(self, lhs, rhs):
        """Swap LHS and RHS and call the normal contains processing."""
        self.lhs = rhs
        self.rhs = lhs
        return super()._preprocess(rhs, lhs)


""" Type dispatch dictionary for intersects binary predicates. """
DispatchDict = {
    (Point, Point): RootIntersects,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): PointPolygonIntersects,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): RootIntersects,
    (Polygon, MultiPoint): RootIntersects,
    (Polygon, LineString): RootIntersects,
    (Polygon, Polygon): RootIntersects,
}
