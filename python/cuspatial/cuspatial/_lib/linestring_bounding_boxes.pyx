# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport data_from_unique_ptr

from cuspatial._lib.cpp.linestring_bounding_box cimport (
    linestring_bounding_boxes as cpp_linestring_bounding_boxes,
)


cpdef linestring_bounding_boxes(Column poly_offsets,
                                Column x, Column y,
                                double R):
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_linestring_bounding_boxes(
            c_poly_offsets, c_x, c_y, R
        ))
    return data_from_unique_ptr(
        move(result),
        column_names=["x_min", "y_min", "x_max", "y_max"]
    )
