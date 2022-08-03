# Copyright (c) 2021-2022 NVIDIA CORPORATION
from typing import Tuple, TypeVar

import cupy as cp
import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase

from cuspatial.geometry.geometa import GeoMeta

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
        from_read_polygon_shapefile=False,
    ):
        if from_read_polygon_shapefile:
            polygons_col = data[0].astype("int32")
            rings_col = data[1].astype("int32")
            coordinates = (
                data[2].stack().astype("float64").reset_index(drop=True)
            )
            coordinate_offsets = cudf.concat(
                [
                    cudf.Series(
                        cp.arange(len(coordinates) // 2) * 2, dtype="int32"
                    ),
                    cudf.Series([len(coordinates) // 2 * 2], dtype="int32"),
                ]
            ).reset_index(drop=True)
            point_offsets = cudf.concat(
                [
                    cudf.Series(
                        cp.arange(len(coordinates) // 2), dtype="int32"
                    ),
                    cudf.Series([len(coordinates) // 2 - 1], dtype="int32"),
                ]
            ).reset_index(drop=True)
            rings_offsets = cudf.concat(
                [
                    cudf.Series(rings_col),
                    cudf.Series([len(point_offsets) - 1], dtype="int32"),
                ]
            ).reset_index(drop=True)
            polygons_offsets = cudf.concat(
                [
                    cudf.Series(polygons_col),
                    cudf.Series([len(polygons_col)], dtype="int32"),
                ]
            ).reset_index(drop=True)
            coords = cudf.core.column.ListColumn(
                size=len(coordinate_offsets) - 1,
                dtype=cudf.ListDtype(coordinates.dtype),
                children=(coordinate_offsets._column, coordinates._column),
            )
            points = cudf.core.column.ListColumn(
                size=len(rings_offsets) - 1,
                dtype=cudf.ListDtype(coords.dtype),
                children=(rings_offsets._column, coords),
            )
            rings = cudf.core.column.ListColumn(
                size=len(polygons_offsets) - 1,
                dtype=cudf.ListDtype(points.dtype),
                children=(polygons_offsets._column, points),
            )
            parent = cudf.core.column.ListColumn(
                size=len(polygons_offsets) - 1,
                dtype=cudf.ListDtype(rings.dtype),
                children=(polygons_offsets._column, rings),
            )
            self.points = cudf.Series([])
            self.points.name = "points"
            self.mpoints = cudf.Series([])
            self.mpoints.name = "mpoints"
            self.lines = cudf.Series([])
            self.lines.name = "lines"
            self.polygons = cudf.Series(parent)
            self.polygons.name = "polygons"
            self._meta = meta

        elif (
            isinstance(data[0], cudf.Series)
            and isinstance(data[1], cudf.Series)
            and isinstance(data[2], cudf.Series)
            and isinstance(data[3], cudf.Series)
        ):
            self._meta = GeoMeta(meta)
            self.points = data[0]
            self.points.name = "points"
            self.mpoints = data[1]
            self.mpoints.name = "mpoints"
            self.lines = data[2]
            self.lines.name = "lines"
            self.polygons = data[3]
            self.polygons.name = "polygons"
        else:
            raise TypeError("All four Tuple arguments must be cudf.ListSeries")
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
