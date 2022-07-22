# Copyright (c) 2022 NVIDIA CORPORATION

from typing import List

import pyarrow as pa

ArrowPolygonsType: pa.ListType = pa.list_(
    pa.list_(pa.list_(pa.list_(pa.float64())))
)

ArrowLinestringsType: pa.ListType = pa.list_(pa.list_(pa.list_(pa.float64())))

ArrowMultiPointsType: pa.ListType = pa.list_(pa.list_(pa.float64()))

ArrowPointsType: pa.ListType = pa.list_(pa.float64())


def getGeoArrowUnionRootType() -> pa.union:
    return pa.union(
        [
            ArrowPointsType,
            ArrowMultiPointsType,
            ArrowLinestringsType,
            ArrowPolygonsType,
        ],
        mode="dense",
    )


def from_lists(
    type_buffer: List,
    all_offsets: List,
    point_coords: List,
    mpoint_coords: List,
    line_coords: List,
    polygon_coords: List,
) -> pa.lib.UnionArray:
    type_buffer = pa.array(type_buffer).cast(pa.int8())
    all_offsets = pa.array(all_offsets).cast(pa.int32())
    children = [
        pa.array(point_coords, type=ArrowPointsType),
        pa.array(mpoint_coords, type=ArrowMultiPointsType),
        pa.array(line_coords, type=ArrowLinestringsType),
        pa.array(polygon_coords, type=ArrowPolygonsType),
    ]

    return pa.UnionArray.from_dense(
        type_buffer,
        all_offsets,
        children,
        ["points", "mpoints", "lines", "polygons"],
    )
