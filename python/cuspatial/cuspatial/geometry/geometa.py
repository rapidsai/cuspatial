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
    Creates input_types and input_lengths for GeoColumns that are created
    using native GeoArrowBuffers. These will be used to convert to GeoPandas
    GeoSeries if necessary.
    """

    def __init__(self, meta: dict):
        if isinstance(meta, dict):
            self.input_types = cudf.Series(meta["input_types"])
            self.input_lengths = cudf.Series(meta["input_lengths"])
            return
        buffers = meta
        self.input_types = []
        self.input_lengths = []
        if buffers.points is not None:
            self.input_types.extend(repeat(0, len(buffers.points)))
            self.input_lengths.extend(repeat(1, len(buffers.points)))
        if buffers.multipoints is not None:
            self.input_types.extend(repeat(1, len(buffers.multipoints)))
            self.input_lengths.extend(repeat(1, len(buffers.multipoints)))
        if buffers.lines is not None:
            for index in range(len(buffers.lines.mlines) - 1):
                line_len = (
                    buffers.lines.mlines[index + 1]
                    - buffers.lines.mlines[index]
                )
                if line_len > 1:
                    self.input_types.extend([3])
                    self.input_lengths.extend([line_len])
                else:
                    self.input_types.extend([2])
                    self.input_lengths.extend([1])
        if buffers.polygons is not None:
            for index in range(len(buffers.polygons.mpolys) - 1):
                poly_len = (
                    buffers.polygons.mpolys[index + 1]
                    - buffers.polygons.mpolys[index]
                )
                if poly_len > 1:
                    self.input_types.extend([5])
                    self.input_lengths.extend([poly_len])
                else:
                    self.input_types.extend([4])
                    self.input_lengths.extend([1])
        self.input_types = cudf.Series(self.input_types, type=pa.int8())
        self.input_lengths = cudf.Series(self.input_lengths).cast(pa.int32())

    def copy(self):
        return self.__class__(
            {
                "input_types": self.input_types.copy(),
                "input_lengths": self.input_lengths.copy(),
            }
        )
