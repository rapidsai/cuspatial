# Copyright (c) 2021-2022 NVIDIA CORPORATION
from functools import cached_property
from typing import Tuple, TypeVar

import cupy as cp
import pyarrow as pa

import cudf
from cudf.core.column import ColumnBase, arange, as_column, build_list_column

from cuspatial.core._column.geometa import Feature_Enum, GeoMeta
from cuspatial.utils.column_utils import empty_geometry_column

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

    @property
    def valid_count(self) -> int:
        """
        Arrow's UnionArray does not support nulls, so this is always
        equal to the length of the GeoColumn.
        """
        return self._meta.input_types.valid_count

    def has_nulls(self) -> bool:
        """
        Arrow's UnionArray does not support nulls, so this is always
        False.
        """
        return self._meta.input_types.has_nulls

    @classmethod
    def _from_points_xy(cls, points_xy: ColumnBase):
        """
        Create a GeoColumn of only single points from a cudf Series with
        interleaved xy coordinates.
        """
        if not points_xy.dtype.kind == "f":
            raise ValueError("Coordinates must be floating point numbers.")

        point_col = _xy_as_variable_sized_list(points_xy)
        num_points = len(point_col)

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

        coord_dtype = points_xy.dtype
        return cls(
            (
                cudf.Series(point_col),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.MULTIPOINT, coord_dtype)
                ),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.LINESTRING, coord_dtype)
                ),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.POLYGON, coord_dtype)
                ),
            ),
            meta,
        )

    @classmethod
    def _from_multipoints_xy(
        cls, multipoints_xy: ColumnBase, geometry_offsets: ColumnBase
    ):
        """
        Create a GeoColumn of multipoints from a cudf Series with
        interleaved xy coordinates.
        """
        if not multipoints_xy.dtype.kind == "f":
            raise ValueError("Coordinates must be floating point numbers.")

        multipoint_col = build_list_column(
            indices=geometry_offsets,
            elements=_xy_as_variable_sized_list(multipoints_xy),
            size=len(geometry_offsets) - 1,
        )
        num_multipoints = len(multipoint_col)

        meta = GeoMeta(
            {
                "input_types": as_column(
                    cp.full(
                        num_multipoints,
                        Feature_Enum.MULTIPOINT.value,
                        dtype=cp.int8,
                    )
                ),
                "union_offsets": as_column(
                    cp.arange(num_multipoints, dtype=cp.int32)
                ),
            }
        )

        coord_dtype = multipoints_xy.dtype

        return cls(
            (
                cudf.Series(
                    empty_geometry_column(Feature_Enum.POINT, coord_dtype)
                ),
                cudf.Series(multipoint_col),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.LINESTRING, coord_dtype)
                ),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.POLYGON, coord_dtype)
                ),
            ),
            meta,
        )

    @classmethod
    def _from_linestrings_xy(
        cls,
        linestrings_xy: ColumnBase,
        part_offsets: ColumnBase,
        geometry_offsets: ColumnBase,
    ):
        """
        Create a GeoColumn of multilinestrings from a cudf Series with
        interleaved xy coordinates.
        """
        if not linestrings_xy.dtype.kind == "f":
            raise ValueError("Coordinates must be floating point numbers.")

        parts_col = build_list_column(
            indices=part_offsets,
            elements=_xy_as_variable_sized_list(linestrings_xy),
            size=len(part_offsets) - 1,
        )
        linestrings_col = build_list_column(
            indices=geometry_offsets,
            elements=parts_col,
            size=len(geometry_offsets) - 1,
        )
        num_linestrings = len(linestrings_col)

        meta = GeoMeta(
            {
                "input_types": as_column(
                    cp.full(
                        num_linestrings,
                        Feature_Enum.LINESTRING.value,
                        dtype=cp.int8,
                    )
                ),
                "union_offsets": as_column(
                    cp.arange(num_linestrings, dtype=cp.int32)
                ),
            }
        )

        coord_dtype = linestrings_xy.dtype

        return cls(
            (
                cudf.Series(
                    empty_geometry_column(Feature_Enum.POINT, coord_dtype)
                ),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.MULTIPOINT, coord_dtype)
                ),
                cudf.Series(linestrings_col),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.POLYGON, coord_dtype)
                ),
            ),
            meta,
        )

    @classmethod
    def _from_polygons_xy(
        cls,
        polygons_xy: ColumnBase,
        ring_offsets: ColumnBase,
        part_offsets: ColumnBase,
        geometry_offsets: ColumnBase,
    ):
        """
        Create a GeoColumn of multipolygons from a cudf Series with
        interleaved xy coordinates.
        """
        if not polygons_xy.dtype.kind == "f":
            raise ValueError("Coordinates must be floating point numbers.")

        rings_col = build_list_column(
            indices=ring_offsets,
            elements=_xy_as_variable_sized_list(polygons_xy),
            size=len(ring_offsets) - 1,
        )
        parts_col = build_list_column(
            indices=part_offsets,
            elements=rings_col,
            size=len(part_offsets) - 1,
        )
        polygons_col = build_list_column(
            indices=geometry_offsets,
            elements=parts_col,
            size=len(geometry_offsets) - 1,
        )
        num_polygons = len(polygons_col)

        meta = GeoMeta(
            {
                "input_types": as_column(
                    cp.full(
                        num_polygons,
                        Feature_Enum.POLYGON.value,
                        dtype=cp.int8,
                    )
                ),
                "union_offsets": as_column(
                    cp.arange(num_polygons, dtype=cp.int32)
                ),
            }
        )

        coord_dtype = polygons_xy.dtype

        return cls(
            (
                cudf.Series(
                    empty_geometry_column(Feature_Enum.POINT, coord_dtype)
                ),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.MULTIPOINT, coord_dtype)
                ),
                cudf.Series(
                    empty_geometry_column(Feature_Enum.LINESTRING, coord_dtype)
                ),
                cudf.Series(polygons_col),
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


def _xy_as_variable_sized_list(xy: ColumnBase):
    """Given an array of interleaved x-y coordinate, construct a cuDF ListDtype
    type array, where each row is the coordinate.
    """
    if len(xy) % 2 != 0:
        raise ValueError("xy must have an even number of elements")

    num_points = len(xy) // 2
    indices = arange(0, num_points * 2 + 1, 2, dtype="int32")
    return build_list_column(indices=indices, elements=xy, size=num_points)
