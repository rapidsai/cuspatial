from typing import Optional

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.distance.point_distance cimport (
    pairwise_point_distance as c_pairwise_point_distance,
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
