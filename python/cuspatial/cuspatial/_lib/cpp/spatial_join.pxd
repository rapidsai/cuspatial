# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column_view
from cudf._lib.cpp.table.table cimport table, table_view


cdef extern from "cuspatial/spatial_join.hpp" namespace "cuspatial" nogil:

    cdef unique_ptr[table] join_quadtree_and_bounding_boxes \
        "cuspatial::join_quadtree_and_bounding_boxes" (
        const table_view & quadtree,
        const table_view & bboxes,
        double x_min,
        double x_max,
        double y_min,
        double y_max,
        double scale,
        int8_t max_depth
    ) except +

    cdef unique_ptr[table] quadtree_point_in_polygon \
        "cuspatial::quadtree_point_in_polygon" (
        const table_view & poly_quad_pairs,
        const table_view & quadtree,
        const column_view & point_indices,
        const column_view & points_x,
        const column_view & points_y,
        const column_view & poly_offsets,
        const column_view & ring_offsets,
        const column_view & poly_points_x,
        const column_view & poly_points_y
    ) except +

    cdef unique_ptr[table] quadtree_point_to_nearest_linestring \
        "cuspatial::quadtree_point_to_nearest_linestring" (
        const table_view & linestring_quad_pairs,
        const table_view & quadtree,
        const column_view & point_indices,
        const column_view & points_x,
        const column_view & points_y,
        const column_view & linestring_offsets,
        const column_view & linestring_points_x,
        const column_view & linestring_points_y
    ) except +
