# Copyright (c) 2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view

from cuspatial._lib.cpp.optional cimport optional


cdef extern from "cuspatial/point_linestring_nearest_points.hpp" \
        namespace "cuspatial" nogil:

    cdef struct point_linestring_nearest_points_result:
        optional[unique_ptr[column]] nearest_point_geometry_id
        optional[unique_ptr[column]] nearest_linestring_geometry_id
        unique_ptr[column] nearest_segment_id
        unique_ptr[column] nearest_point_on_linestring_xy

    cdef point_linestring_nearest_points_result \
        pairwise_point_linestring_nearest_points(
            const optional[column_view] multipoint_geometry_offsets,
            const column_view points_xy,
            const optional[column_view] multilinestring_geometry_offsets,
            const column_view linestring_part_offsets,
            const column_view linestring_points_xy,
        ) except +
