# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import (
    DispatchDict as CONTAINS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_covers import (
    DispatchDict as COVERS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_equals import (
    DispatchDict as EQUALS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_intersects import (
    DispatchDict as INTERSECTS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_within import RootWithin

Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

CONTAINS_DISPATCH = CONTAINS_DISPATCH_DICT

INTERSECTS_DISPATCH = INTERSECTS_DISPATCH_DICT

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

EQUALS_DISPATCH = EQUALS_DISPATCH_DICT

COVERS_DISPATCH = COVERS_DISPATCH_DICT
