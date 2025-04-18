# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuspatial._lib.cpp.point_in_polygon cimport (
    point_in_polygon as cpp_point_in_polygon,
)


def point_in_polygon(
    plc_Column test_points_x,
    plc_Column test_points_y,
    plc_Column poly_offsets,
    plc_Column poly_ring_offsets,
    plc_Column poly_points_x,
    plc_Column poly_points_y
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

    return plc_Column.from_libcudf(move(result))
