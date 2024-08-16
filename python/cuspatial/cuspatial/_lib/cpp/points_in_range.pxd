# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view


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
