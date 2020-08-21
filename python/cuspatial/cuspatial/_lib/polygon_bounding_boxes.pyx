# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cuspatial._lib.cpp.polygon_bounding_box cimport (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)

from cuspatial._lib.move cimport move

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

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
    return Table.from_unique_ptr(
        move(result),
        column_names=["x_min", "y_min", "x_max", "y_max"]
    )
