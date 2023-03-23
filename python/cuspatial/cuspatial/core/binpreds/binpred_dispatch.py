# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from cudf.core.series import Series

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import (
    PointPointContains,
    PolygonLineStringContains,
    PolygonMultiPointContains,
    PolygonPointContains,
    PolygonPolygonContains,
)

Point = ColumnType.Point
MultiPoint = ColumnType.MultiPoint
LineString = ColumnType.LineString
Polygon = ColumnType.Polygon

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
