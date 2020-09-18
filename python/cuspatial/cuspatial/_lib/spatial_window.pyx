# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.table cimport table, Table
from cudf._lib.column cimport column, column_view, Column

from cuspatial._lib.cpp.spatial_window \
    cimport points_in_spatial_window as cpp_points_in_spatial_window

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr

cpdef points_in_spatial_window(
    double window_min_x,
    double window_max_x,
    double window_min_y,
    double window_max_y,
    Column x,
    Column y
):
    cdef column_view x_v = x.view()
    cdef column_view y_v = y.view()

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_points_in_spatial_window(
                window_min_x,
                window_max_x,
                window_min_y,
                window_max_y,
                x_v,
                y_v
            )
        )

    table = Table.from_unique_ptr(move(c_result), column_names=["x", "y"])
    return table
