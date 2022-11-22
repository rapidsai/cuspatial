# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type
from cudf._lib.utils cimport data_from_unique_ptr

from cuspatial._lib.cpp.quadtree cimport (
    quadtree_on_points as cpp_quadtree_on_points,
)


cpdef quadtree_on_points(Column x, Column y,
                         double x_min, double x_max,
                         double y_min, double y_max,
                         double scale,
                         int8_t max_depth,
                         size_type min_size):
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef pair[unique_ptr[column], unique_ptr[table]] result
    with nogil:
        result = move(cpp_quadtree_on_points(
            c_x, c_y, x_min, x_max, y_min, y_max, scale, max_depth, min_size
        ))
    return (
        Column.from_unique_ptr(move(result.first)),
        data_from_unique_ptr(
            move(result.second),
            column_names=[
                "key",
                "level",
                "is_internal_node",
                "length",
                "offset"
            ]
        )
    )
