# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.column.column cimport column

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)


cdef extern from "cuspatial/distance.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] pairwise_polygon_distance(
        const geometry_column_view & lhs,
        const geometry_column_view & rhs
    ) except +
