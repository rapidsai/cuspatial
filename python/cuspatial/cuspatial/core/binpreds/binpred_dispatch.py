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
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.core.binpreds.feature_intersects import RootIntersects
from cuspatial.core.binpreds.feature_within import RootWithin

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

INTERSECTS_DISPATCH = {
    (Point, Point): RootIntersects,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootIntersects,
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

WITHIN_DISPATCH = {
    (Point, Point): RootWithin,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootWithin,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): RootWithin,
    (Polygon, Point): RootWithin,
    (Polygon, MultiPoint): RootWithin,
    (Polygon, LineString): RootWithin,
    (Polygon, Polygon): RootWithin,
}

EQUALS_DISPATCH = {
    (Point, Point): RootEquals,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootEquals,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): RootEquals,
    (Polygon, Point): RootEquals,
    (Polygon, MultiPoint): RootEquals,
    (Polygon, LineString): RootEquals,
    (Polygon, Polygon): RootEquals,
}
