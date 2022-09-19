# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.tuple cimport 

from cudf._lib.column cimport column, column_view
from cuspatial._lib.cpp.optional cimport optional

cdef extern from "cuspatial/point_linestring_nearest_point.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[column] pairwise_point_linestring_nearest_point(
        const optional[column_view] multipoint_geometry_offsets,
        const column_view points_xy,
        const optional[column_view] multilinestring_geometry_offsets,
        const column_view linestring_part_offsets,
        const column_view linestring_points_xy,
    ) except +
