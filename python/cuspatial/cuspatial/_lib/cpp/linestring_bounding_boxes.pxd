# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table


cdef extern from "cuspatial/bounding_boxes.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[table] linestring_bounding_boxes(
        const column_view & linestring_offsets,
        const column_view & x,
        const column_view & y,
        double R
    ) except +
