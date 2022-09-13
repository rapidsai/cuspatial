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


def pairwise_point_linestring_distance(
    Column points_xy,
    Column linestring_part_offsets,
    Column linestring_points_xy,
    optional_multipoint_geometry_offset=None,
    optional_multilinestring_geometry_offset=None,
):
    cdef Column multipoint_parts_offset
    cdef Column multilinestring_parts_offset
    cdef optional[column_view] c_multipoint_parts_offset
    cdef optional[column_view] c_multilinestring_parts_offset

    if optional_multipoint_geometry_offset is not None:
        multipoint_parts_offset = optional_multipoint_geometry_offset
        c_multipoint_parts_offset = multipoint_parts_offset.view()
    else:
        c_multipoint_parts_offset = nullopt

    if optional_multilinestring_geometry_offset is not None:
        multilinestring_parts_offset = optional_multilinestring_geometry_offset
        c_multilinestring_parts_offset = multilinestring_parts_offset.view()
    else:
        c_multilinestring_parts_offset = nullopt

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
