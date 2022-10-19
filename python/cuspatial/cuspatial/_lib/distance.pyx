from typing import Optional

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.distance.linestring_distance cimport (
    pairwise_linestring_distance as c_pairwise_linestring_distance,
)
from cuspatial._lib.cpp.distance.point_distance cimport (
    pairwise_point_distance as c_pairwise_point_distance,
)
from cuspatial._lib.cpp.distance.point_linestring_distance cimport (
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
)
from cuspatial._lib.cpp.optional cimport nullopt, optional
from cuspatial._lib.utils cimport unwrap_pyoptcol


def pairwise_point_distance(
    Column points1_xy,
    Column points2_xy,
    multipoint1_offsets=None,
    multipoint2_offsets=None,
):
    cdef optional[column_view] c_multipoints1_offset = unwrap_pyoptcol(
        multipoint1_offsets)
    cdef optional[column_view] c_multipoints2_offset = unwrap_pyoptcol(
        multipoint2_offsets)

    cdef column_view c_points1_xy = points1_xy.view()
    cdef column_view c_points2_xy = points2_xy.view()
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(c_pairwise_point_distance(
            c_multipoints1_offset,
            c_points1_xy,
            c_multipoints2_offset,
            c_points2_xy,
        ))
    return Column.from_unique_ptr(move(c_result))


def pairwise_linestring_distance(
    Column linestring1_offsets,
    Column linestring1_points_x,
    Column linestring1_points_y,
    Column linestring2_offsets,
    Column linestring2_points_x,
    Column linestring2_points_y
):
    cdef column_view linestring1_offsets_view = linestring1_offsets.view()
    cdef column_view linestring1_points_x_view = linestring1_points_x.view()
    cdef column_view linestring1_points_y_view = linestring1_points_y.view()
    cdef column_view linestring2_offsets_view = linestring2_offsets.view()
    cdef column_view linestring2_points_x_view = linestring2_points_x.view()
    cdef column_view linestring2_points_y_view = linestring2_points_y.view()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(c_pairwise_linestring_distance(
            linestring1_offsets_view,
            linestring1_points_x_view,
            linestring1_points_y_view,
            linestring2_offsets_view,
            linestring2_points_x_view,
            linestring2_points_y_view
        ))

    return Column.from_unique_ptr(move(c_result))


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
