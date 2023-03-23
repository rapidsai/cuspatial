# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import (
    PointPointContains,
    PolygonLineStringContains,
    PolygonMultiPointContains,
    PolygonPointContains,
    PolygonPolygonContains,
)

Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

CONTAINS_DISPATCH = {
    (Point, Point): PointPointContains,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): NotImplementedRoot,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): PolygonPointContains,
    (Polygon, MultiPoint): PolygonMultiPointContains,
    (Polygon, LineString): PolygonLineStringContains,
    (Polygon, Polygon): PolygonPolygonContains,
}
