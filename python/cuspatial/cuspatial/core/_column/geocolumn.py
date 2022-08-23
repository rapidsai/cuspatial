# Copyright (c) 2021-2022 NVIDIA CORPORATION
from typing import Tuple, TypeVar

import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase

from cuspatial.core._column.geometa import GeoMeta

T = TypeVar("T", bound="GeoColumn")


class GeoColumn(ColumnBase):
    """
    Parameters
    ----------
    data : A tuple of four cudf.Series of list dtype
    meta : A GeoMeta object (optional)

    Notes
    -----
    The GeoColumn class subclasses `NumericalColumn`. Combined with
    `_copy_type_metadata`, this assures support for sort, groupby,
    and potential other `cudf` algorithms.
    """

    def __init__(
        self,
        data: Tuple,
        meta: GeoMeta = None,
        shuffle_order: cudf.Index = None,
    ):
        if (
            not isinstance(data[0], cudf.Series)
            or not isinstance(data[1], cudf.Series)
            or not isinstance(data[2], cudf.Series)
            or not isinstance(data[3], cudf.Series)
        ):
            raise TypeError("All Tuple arguments must be cudf.ListSeries")
        self._meta = GeoMeta(meta)
        self.points = data[0]
        self.points.name = "points"
        self.mpoints = data[1]
        self.mpoints.name = "mpoints"
        self.lines = data[2]
        self.lines.name = "lines"
        self.polygons = data[3]
        self.polygons.name = "polygons"
        base = cudf.core.column.column.arange(0, len(self), dtype="int32").data
        super().__init__(base, size=len(self), dtype="int32")
        if shuffle_order is not None:
            self._data = shuffle_order

    def to_arrow(self):
        return pa.UnionArray.from_dense(
            self._meta.input_types.to_arrow(),
            self._meta.union_offsets.to_arrow(),
            [
                self.points.to_arrow(),
                self.mpoints.to_arrow(),
                self.lines.to_arrow(),
                self.polygons.to_arrow(),
            ],
        )

    def __len__(self):
        """
        Returns the number of unique geometries stored in this GeoColumn.
        """
        return len(self._meta.input_types)

    def _dump(self):
        return (
            f"POINTS\n"
            f"{self.points._repr__()}\n"
            f"MULTIPOINTS\n"
            f"{self.multipoints._repr__()}\n"
            f"LINES\n"
            f"{self.lines._repr__()}\n"
            f"POLYGONS\n"
            f"{self.polygons._repr__()}\n"
        )

    def copy(self, deep=True):
        """
        Create a copy of all of the GPU-backed data structures in this
        GeoColumn.
        """
        result = GeoColumn(
            (
                self.points.copy(deep),
                self.mpoints.copy(deep),
                self.lines.copy(deep),
                self.polygons.copy(deep),
            ),
            self._meta.copy(),
            self.data.copy(),
        )
        return result
