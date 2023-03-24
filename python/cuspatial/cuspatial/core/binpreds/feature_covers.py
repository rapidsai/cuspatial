# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_equals import RootEquals


class RootCovers(RootEquals):
    pass


Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

DispatchDict = {
    (Point, Point): RootCovers,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootCovers,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): RootCovers,
    (Polygon, Point): RootCovers,
    (Polygon, MultiPoint): RootCovers,
    (Polygon, LineString): RootCovers,
    (Polygon, Polygon): RootCovers,
}
