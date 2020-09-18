# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.column cimport column, column_view, Column

from cuspatial._lib.cpp.point_in_polygon cimport (
    point_in_polygon as cpp_point_in_polygon,
)

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr


def point_in_polygon(
    Column test_points_x,
    Column test_points_y,
    Column poly_offsets,
    Column poly_ring_offsets,
    Column poly_points_x,
    Column poly_points_y
):
    cdef column_view c_test_points_x = test_points_x.view()
    cdef column_view c_test_points_y = test_points_y.view()
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_poly_ring_offsets = poly_ring_offsets.view()
    cdef column_view c_poly_points_x = poly_points_x.view()
    cdef column_view c_poly_points_y = poly_points_y.view()

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_point_in_polygon(
                c_test_points_x,
                c_test_points_y,
                c_poly_offsets,
                c_poly_ring_offsets,
                c_poly_points_x,
                c_poly_points_y
            )
        )

    return Column.from_unique_ptr(move(result))
