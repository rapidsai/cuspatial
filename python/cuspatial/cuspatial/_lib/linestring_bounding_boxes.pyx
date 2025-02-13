# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column, Table as plc_Table
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table

from cuspatial._lib.cpp.linestring_bounding_boxes cimport (
    linestring_bounding_boxes as cpp_linestring_bounding_boxes,
)


cpdef list linestring_bounding_boxes(
    plc_Column poly_offsets,
    plc_Column x,
    plc_Column y,
    double R
):
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_linestring_bounding_boxes(
            c_poly_offsets, c_x, c_y, R
        ))
    cdef plc_Table plc_table = plc_Table.from_libcudf(move(result))
    return plc_table.columns()
