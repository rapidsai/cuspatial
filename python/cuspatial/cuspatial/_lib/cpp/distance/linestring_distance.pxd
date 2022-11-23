# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from cuspatial._lib.cpp.optional cimport optional


cdef extern from "cuspatial/distance/linestring_distance.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] pairwise_linestring_distance(
        const optional[column_view] multilinestring1_geometry_offsets,
        const column_view linestring1_part_offsets,
        const column_view linestring1_points_xy,
        const optional[column_view] multilinestring2_geometry_offsets,
        const column_view linestring2_part_offsets,
        const column_view linestring2_points_xy
    ) except +
