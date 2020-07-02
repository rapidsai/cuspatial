# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from libcpp.memory cimport unique_ptr

cdef extern from "cuspatial/polyline_bounding_box.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[table] polyline_bounding_boxes(
        const column_view & poly_offsets,
        const column_view & x,
        const column_view & y,
        double R
    ) except +
