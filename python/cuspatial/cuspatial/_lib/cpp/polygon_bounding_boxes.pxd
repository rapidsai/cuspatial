# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table


cdef extern from "cuspatial/bounding_boxes.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[table] polygon_bounding_boxes(
        const column_view & poly_offsets,
        const column_view & ring_offsets,
        const column_view & x,
        const column_view & y
    ) except +
