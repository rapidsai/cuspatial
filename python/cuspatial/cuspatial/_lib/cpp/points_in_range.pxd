# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table, table_view


cdef extern from "cuspatial/points_in_range.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] points_in_range \
        "cuspatial::points_in_range" (
        double range_min_x,
        double range_max_x,
        double range_min_y,
        double range_max_y,
        const column_view & x,
        const column_view & y
    ) except +
