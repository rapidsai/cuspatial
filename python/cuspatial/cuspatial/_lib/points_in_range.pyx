# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.utils cimport data_from_unique_ptr
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table

from cuspatial._lib.cpp.points_in_range cimport (
    points_in_range as cpp_points_in_range,
)


cpdef points_in_range(
    double range_min_x,
    double range_max_x,
    double range_min_y,
    double range_max_y,
    Column x,
    Column y
):
    cdef column_view x_v = x.view()
    cdef column_view y_v = y.view()

    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(
            cpp_points_in_range(
                range_min_x,
                range_max_x,
                range_min_y,
                range_max_y,
                x_v,
                y_v
            )
        )

    return data_from_unique_ptr(move(c_result), column_names=["x", "y"])
