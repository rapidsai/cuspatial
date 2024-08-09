# Copyright (c) 2021-2024, NVIDIA CORPORATION

# This allows GeoMeta as its own init type
from __future__ import annotations

from enum import Enum
from typing import Literal, Union

import cudf
import cudf.core.column


# This causes arrow to encode NONE as =255, which I'll accept now
# in order to keep the rest of the enums the same.
class Feature_Enum(Enum):
    NONE = -1
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

    def __init__(
        self,
        meta: Union[
            GeoMeta,
            dict[
                Literal["input_types", "union_offsets"],
                cudf.core.column.ColumnBase,
            ],
        ],
    ):
        if isinstance(meta, dict):
            meta_it = meta["input_types"]
            if isinstance(meta_it, cudf.core.column.ColumnBase):
                self.input_types = cudf.Series._from_column(meta_it).astype(
                    "int8"
                )
            else:
                # Could be Series from GeoSeries.__getitem__
                self.input_types = cudf.Series(meta_it, dtype="int8")
            meta_uo = meta["union_offsets"]
            if isinstance(meta_uo, cudf.core.column.ColumnBase):
                self.union_offsets = cudf.Series._from_column(meta_uo).astype(
                    "int32"
                )
            else:
                # Could be Series from GeoSeries.__getitem__
                self.union_offsets = cudf.Series(meta_uo, dtype="int32")
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
