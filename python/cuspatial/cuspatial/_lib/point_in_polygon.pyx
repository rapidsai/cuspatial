# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column, column, column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_table_view

from cuspatial._lib.cpp.point_in_polygon cimport (
    byte_point_in_polygon as cpp_byte_point_in_polygon,
    columnar_point_in_polygon as cpp_columnar_point_in_polygon,
)


def byte_point_in_polygon(
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
            cpp_byte_point_in_polygon(
                c_test_points_x,
                c_test_points_y,
                c_poly_offsets,
                c_poly_ring_offsets,
                c_poly_points_x,
                c_poly_points_y
            )
        )

    return Column.from_unique_ptr(move(result))


def columnar_point_in_polygon(
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

    cdef pair[unique_ptr[column], table_view] result

    with nogil:
        result = move(
            cpp_columnar_point_in_polygon(
                c_test_points_x,
                c_test_points_y,
                c_poly_offsets,
                c_poly_ring_offsets,
                c_poly_points_x,
                c_poly_points_y
            )
        )

    result_owner = Column.from_unique_ptr(move(result.first))
    return columns_from_table_view(
        result.second, owners=[result_owner] * result.second.num_columns())
