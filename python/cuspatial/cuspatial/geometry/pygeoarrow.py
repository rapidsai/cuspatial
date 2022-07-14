# Copyright (c) 2021 NVIDIA CORPORATION

from re import L
from typing import Tuple
import pyarrow as pa
import geopandas as gpd


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


def from_geopandas(buffers: dict) -> pa.lib.UnionArray:
    type_buffer = pa.array(buffers["type_buffer"]).cast(pa.int8())
    all_offsets = pa.array(buffers["all_offsets"]).cast(pa.int32())
    children = [
        pa.array(buffers["point_coords"], type=ArrowPointsType),
        pa.array(buffers["mpoint_coords"], type=ArrowMultiPointsType),
        pa.array(buffers["line_coords"], type=ArrowLinestringsType),
        pa.array(buffers["polygon_coords"], type=ArrowPolygonsType),
    ]

    return pa.UnionArray.from_dense(
        type_buffer,
        all_offsets,
        children,
        ["points", "mpoints", "lines", "polygons"],
    )
