from typing import Optional

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.distance.point_linestring_distance cimport (
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
)
from cuspatial._lib.cpp.optional cimport nullopt, optional
from cuspatial._lib.utils cimport unwrap_pyoptcol


def pairwise_point_linestring_distance(
    Column points_xy,
    Column linestring_part_offsets,
    Column linestring_points_xy,
    multipoint_geometry_offset=None,
    multilinestring_geometry_offset=None,
):
    cdef optional[column_view] c_multipoint_parts_offset = unwrap_pyoptcol(
        multipoint_geometry_offset)
    cdef optional[column_view] c_multilinestring_parts_offset = (
        unwrap_pyoptcol(multilinestring_geometry_offset))

    cdef column_view c_points_xy = points_xy.view()
    cdef column_view c_linestring_offsets = linestring_part_offsets.view()
    cdef column_view c_linestring_points_xy = linestring_points_xy.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_linestring_distance(
            c_multipoint_parts_offset,
            c_points_xy,
            c_multilinestring_parts_offset,
            c_linestring_offsets,
            c_linestring_points_xy,
        ))
    return Column.from_unique_ptr(move(c_result))
