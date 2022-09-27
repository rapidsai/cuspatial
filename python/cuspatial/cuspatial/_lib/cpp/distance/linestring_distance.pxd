# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type


cdef extern from "cuspatial/distance/linestring_distance.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] pairwise_linestring_distance(
        const column_view linestring1_offsets,
        const column_view linestring1_points_x,
        const column_view linestring1_points_y,
        const column_view linestring2_offsets,
        const column_view linestring2_points_x,
        const column_view linestring2_points_y
    ) except +
