# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)


cdef extern from "cuspatial/pairwise_multipoint_equals_count.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] pairwise_multipoint_equals_count(
        const geometry_column_view lhs,
        const geometry_column_view rhs,
    ) except +
