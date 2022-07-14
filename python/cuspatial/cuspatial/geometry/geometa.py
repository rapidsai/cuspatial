# Copyright (c) 2021-2022 NVIDIA CORPORATION

from typing import Union

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

    def __init__(self, meta: Union[GeoMeta, dict]):
        if isinstance(meta, dict):
            self.input_types = cudf.Series(meta["input_types"])
            self.union_offsets = cudf.Series(meta["union_offsets"])
        else:
            self.input_types = cudf.Series(self.input_types, type=pa.int8())
            self.union_offsets = cudf.Series(self.union_offsets).cast(
                pa.int32()
            )

    def copy(self):
        return self.__class__(
            {
                "input_types": self.input_types.copy(),
                "union_offsets": self.union_offsets.copy(),
            }
        )
