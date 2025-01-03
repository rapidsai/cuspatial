# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from pylibcudf cimport Table as plc_Table
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table

from cuspatial._lib.cpp.linestring_bounding_boxes cimport (
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
    cdef plc_Table plc_table = plc_Table.from_libcudf(move(result))
    return [
        Column.from_pylibcudf(col)
        for col in plc_table.columns()
    ]
