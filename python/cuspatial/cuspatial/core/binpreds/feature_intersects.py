# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals


class RootIntersects(RootEquals):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Intersects
    predicate is defined in terms of a Point-Point Intersects predicate.
    """

    pass


class PointPolygonIntersects(RootContains):
    def _preprocess(self, lhs, rhs):
        self.lhs = rhs
        self.rhs = lhs
        return super()._preprocess(rhs, lhs)


Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

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
