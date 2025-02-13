# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column, Table as plc_Table
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type

from cuspatial._lib.cpp.quadtree cimport (
    quadtree_on_points as cpp_quadtree_on_points,
)


cpdef tuple quadtree_on_points(
    plc_Column x,
    plc_Column y,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    double scale,
    int8_t max_depth,
    size_type min_size,
):
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef pair[unique_ptr[column], unique_ptr[table]] result
    with nogil:
        result = move(cpp_quadtree_on_points(
            c_x, c_y, x_min, x_max, y_min, y_max, scale, max_depth, min_size
        ))
    return (
        plc_Column.from_libcudf(move(result.first)),
        plc_Table.from_libcudf(move(result.second))
    )
