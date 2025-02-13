# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column, Table as plc_Table
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table

from cuspatial._lib.cpp.polygon_bounding_boxes cimport (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)


cpdef list polygon_bounding_boxes(
    plc_Column poly_offsets,
    plc_Column ring_offsets,
    plc_Column x,
    plc_Column y,
):
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_ring_offsets = ring_offsets.view()
    cdef column_view c_x = x.view()
    cdef column_view c_y = y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_polygon_bounding_boxes(
            c_poly_offsets, c_ring_offsets, c_x, c_y
        ))
    cdef plc_Table table_result = plc_Table.from_libcudf(move(result))
    return table_result.columns()
