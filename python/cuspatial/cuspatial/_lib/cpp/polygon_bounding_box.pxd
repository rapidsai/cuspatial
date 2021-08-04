# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table


cdef extern from "cuspatial/polygon_bounding_box.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[table] polygon_bounding_boxes(
        const column_view & poly_offsets,
        const column_view & ring_offsets,
        const column_view & x,
        const column_view & y
    ) except +
