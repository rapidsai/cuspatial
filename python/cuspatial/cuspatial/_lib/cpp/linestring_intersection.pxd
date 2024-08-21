# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)


cdef extern from "cuspatial/intersection.hpp" \
        namespace "cuspatial" nogil:

    struct linestring_intersection_column_result:
        unique_ptr[column] geometry_collection_offset

        unique_ptr[column] types_buffer
        unique_ptr[column] offset_buffer

        unique_ptr[column] points

        unique_ptr[column] segments

        unique_ptr[column] lhs_linestring_id
        unique_ptr[column] lhs_segment_id
        unique_ptr[column] rhs_linestring_id
        unique_ptr[column] rhs_segment_id

    cdef linestring_intersection_column_result \
        pairwise_linestring_intersection(
            const geometry_column_view & multilinestring_lhs,
            const geometry_column_view & multilinestring_rhs,
        ) except +
