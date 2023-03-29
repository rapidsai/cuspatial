# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    ImpossibleRoot,
    NotImplementedRoot,
)
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class RootCrosses(RootEquals):
    """Base class for binary predicates that are defined in terms of a
    the equals binary predicate. For example, a Point-Point Crosses
    predicate is defined in terms of a Point-Point Equals predicate.
    """

    pass


class PointPointCrosses(RootCrosses):
    def _preprocess(self, lhs, rhs):
        """Points can't cross other points, so we return False."""
        return _false_series(len(lhs))


DispatchDict = {
    (Point, Point): PointPointCrosses,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootCrosses,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): ImpossibleRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): RootCrosses,
    (Polygon, MultiPoint): RootCrosses,
    (Polygon, LineString): RootCrosses,
    (Polygon, Polygon): RootCrosses,
}
