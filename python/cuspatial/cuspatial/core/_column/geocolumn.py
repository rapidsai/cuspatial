# Copyright (c) 2021-2022 NVIDIA CORPORATION
from functools import cached_property
from typing import Tuple, TypeVar

import cupy as cp
import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase, as_column, build_list_column

from cuspatial.core._column.geometa import Feature_Enum, GeoMeta

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
            """
            A cudf.ListSeries needs four levels of nesting to represent
            a polygon shapefile. The rings_offsets and polygons_offsets
            have already been computed in the `read_polygon_shapefile`
            function. In order to convert it into an arrow list<...>
            we need a set of offsets buffers for each point tuple, and
            an offsets buffer for the 1-offset multipolygons.

            Coordinates: List of length 2 offsets: [0, 2, 4, ... n/2]
            Rings: List of polygon ring offsets
            Polygons: Offset into rings of each polygon
            Multipolygons: List of length 1 offsets: No multipolygons

            Finally, each set of offsets must have the length of the
            array appended to the end, as Arrow offset lists are length
            n + 1 but our original shapefile code offset lists are only
            length n.
            """
            polygons_col = data[0].astype("int32")
            rings_col = data[1].astype("int32")
            coordinates = (
                data[2].stack().astype("float64").reset_index(drop=True)
            )
            """
            Store a fixed-size offsets buffer of even numbers:
            0   0
            1   2
            2   4
            ...
            Up to the size of the original input.
            """
            coordinate_offsets = as_column(
                cp.arange(len(coordinates) + 1, step=2), dtype="int32"
            )
            rings_offsets = cudf.concat(
                [
                    cudf.Series(rings_col),
                    cudf.Series([len(coordinate_offsets) - 1], dtype="int32"),
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
                children=(coordinate_offsets, coordinates._column),
            )
            rings = cudf.core.column.ListColumn(
                size=len(rings_offsets) - 1,
                dtype=cudf.ListDtype(coords.dtype),
                children=(rings_offsets._column, coords),
            )
            polygons = cudf.core.column.ListColumn(
                size=len(polygons_offsets) - 1,
                dtype=cudf.ListDtype(rings.dtype),
                children=(polygons_offsets._column, rings),
            )
            mpolygons = cudf.core.column.ListColumn(
                size=len(polygons_offsets) - 1,
                dtype=cudf.ListDtype(polygons.dtype),
                children=(
                    as_column(cp.arange(len(polygons) + 1), dtype="int32"),
                    polygons,
                ),
            )
            self.points = cudf.Series([])
            self.points.name = "points"
            self.mpoints = cudf.Series([])
            self.mpoints.name = "mpoints"
            self.lines = cudf.Series([])
            self.lines.name = "lines"
            self.polygons = cudf.Series(mpolygons)
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
        super().__init__(None, size=len(self), dtype="geometry")

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
        )
        return result

    @classmethod
    def _from_points_xy(cls, points_xy: ColumnBase):
        """
        Create a GeoColumn of only single points from a cudf Series with
        interleaved xy coordinates.
        """
        if len(points_xy) % 2 != 0:
            raise ValueError("points_xy must have an even number of elements")

        num_points = len(points_xy) // 2
        meta = GeoMeta(
            {
                "input_types": as_column(
                    cp.full(
                        num_points, Feature_Enum.POINT.value, dtype=cp.int8
                    )
                ),
                "union_offsets": as_column(
                    cp.arange(num_points, dtype=cp.int32)
                ),
            }
        )

        indices = as_column(cp.arange(0, num_points * 2 + 1, 2), dtype="int32")
        point_col = build_list_column(
            indices=indices, elements=points_xy, size=num_points
        )
        return cls(
            (
                cudf.Series(point_col),
                cudf.Series(),
                cudf.Series(),
                cudf.Series(),
            ),
            meta,
        )

    @cached_property
    def memory_usage(self) -> int:
        """
        Outputs how much memory is used by the underlying geometries.
        """
        final_size = self._meta.input_types.memory_usage()
        final_size = final_size + self._meta.union_offsets.memory_usage()
        final_size = final_size + self.points._column.memory_usage
        final_size = final_size + self.mpoints._column.memory_usage
        final_size = final_size + self.lines._column.memory_usage
        final_size = final_size + self.polygons._column.memory_usage
        return final_size
