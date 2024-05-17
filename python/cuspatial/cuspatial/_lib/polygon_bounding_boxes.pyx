# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.utils cimport columns_from_unique_ptr

from cuspatial._lib.cpp.polygon_bounding_boxes cimport (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)


cpdef polygon_bounding_boxes(Column poly_offsets,
                             Column ring_offsets,
                             Column x, Column y):
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_ring_offsets = ring_offsets.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_polygon_bounding_boxes(
            c_poly_offsets, c_ring_offsets, c_x, c_y
        ))
    return columns_from_unique_ptr(move(result))
