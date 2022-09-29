# Copyright (c) 2021-2022 NVIDIA CORPORATION

# This allows GeoMeta as its own init type
from __future__ import annotations

from enum import Enum
from typing import Union

import cudf


class Feature_Enum(Enum):
    POINT = 0
    MULTIPOINT = 1
    LINESTRING = 2
    POLYGON = 3


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
            self.input_types = cudf.Series(meta.input_types, dtype="int8")
            self.union_offsets = cudf.Series(meta.union_offsets, dtype="int32")

    def copy(self):
        return self.__class__(
            {
                "input_types": self.input_types.copy(),
                "union_offsets": self.union_offsets.copy(),
            }
        )
