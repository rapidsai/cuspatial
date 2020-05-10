# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport mutable_column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport Column
from cudf._lib.table cimport Table

from cuspatial._lib.cpp.quadtree cimport (
    quadtree_on_points as cpp_quadtree_on_points,
)

from cuspatial._lib.move cimport move

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

cpdef quadtree_on_points(Column x, Column y,
                         double x1, double y1, double x2, double y2,
                         double scale,
                         size_type num_levels,
                         size_type min_size):
    cdef mutable_column_view c_x = x.mutable_view()
    cdef mutable_column_view c_y = y.mutable_view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_quadtree_on_points(
            c_x, c_y, x1, y1, x2, y2, scale, num_levels, min_size
        ))
    return Table.from_unique_ptr(
        move(result),
        column_names=["key", "level", "is_node", "length", "offset"]
    )
