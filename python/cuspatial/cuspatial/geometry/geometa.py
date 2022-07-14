# Copyright (c) 2021-2022 NVIDIA CORPORATION
from itertools import repeat
import pyarrow as pa
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
import cudf


class GeoMeta:
    """
    Creates input_types and union_offsets for GeoColumns that are created
    using native GeoArrowBuffers. These will be used to convert to GeoPandas
    GeoSeries if necessary.
    """

    def __init__(self, meta: dict):
        if isinstance(meta, dict):
            self.input_types = cudf.Series(meta["input_types"])
            self.union_offsets = cudf.Series(meta["union_offsets"])
            return
        buffers = meta
        self.input_types = []
        self.union_offsets = []
        if buffers.points is not None:
            self.input_types.extend(repeat(0, len(buffers.points)))
            self.union_offsets.extend(repeat(1, len(buffers.points)))
        if buffers.multipoints is not None:
            self.input_types.extend(repeat(1, len(buffers.multipoints)))
            self.union_offsets.extend(repeat(1, len(buffers.multipoints)))
        if buffers.lines is not None:
            for index in range(len(buffers.lines.mlines) - 1):
                line_len = (
                    buffers.lines.mlines[index + 1]
                    - buffers.lines.mlines[index]
                )
                if line_len > 1:
                    self.input_types.extend([3])
                    self.union_offsets.extend([line_len])
                else:
                    self.input_types.extend([2])
                    self.union_offsets.extend([1])
        if buffers.polygons is not None:
            for index in range(len(buffers.polygons.mpolys) - 1):
                poly_len = (
                    buffers.polygons.mpolys[index + 1]
                    - buffers.polygons.mpolys[index]
                )
                if poly_len > 1:
                    self.input_types.extend([5])
                    self.union_offsets.extend([poly_len])
                else:
                    self.input_types.extend([4])
                    self.union_offsets.extend([1])
        self.input_types = cudf.Series(self.input_types, type=pa.int8())
        self.union_offsets = cudf.Series(self.union_offsets).cast(pa.int32())

    def copy(self):
        return self.__class__(
            {
                "input_types": self.input_types.copy(),
                "union_offsets": self.union_offsets.copy(),
            }
        )
