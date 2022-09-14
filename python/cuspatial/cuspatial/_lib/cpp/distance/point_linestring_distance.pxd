# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type

from cuspatial._lib.cpp.optional cimport optional


cdef extern from "cuspatial/distance/point_linestring_distance.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] pairwise_point_linestring_distance(
        const optional[column_view] multipoint_geometry_offsets,
        const column_view points_xy,
        const optional[column_view] multilinestring_geometry_offsets,
        const column_view linestring_part_offsets,
        const column_view linestring_points_xy,
    ) except +
