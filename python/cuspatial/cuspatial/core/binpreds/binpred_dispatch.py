# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.feature_contains import (
    DispatchDict as CONTAINS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_covers import (
    DispatchDict as COVERS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_crosses import (
    DispatchDict as CROSSES_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_equals import (
    DispatchDict as EQUALS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_intersects import (
    DispatchDict as INTERSECTS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_overlaps import (
    DispatchDict as OVERLAPS_DISPATCH_DICT,
)
from cuspatial.core.binpreds.feature_within import (
    DispatchDict as WITHIN_DISPATCH_DICT,
)

Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

CONTAINS_DISPATCH = CONTAINS_DISPATCH_DICT

INTERSECTS_DISPATCH = INTERSECTS_DISPATCH_DICT

WITHIN_DISPATCH = WITHIN_DISPATCH_DICT

EQUALS_DISPATCH = EQUALS_DISPATCH_DICT

COVERS_DISPATCH = COVERS_DISPATCH_DICT

OVERLAPS_DISPATCH = OVERLAPS_DISPATCH_DICT

CROSSES_DISPATCH = CROSSES_DISPATCH_DICT
