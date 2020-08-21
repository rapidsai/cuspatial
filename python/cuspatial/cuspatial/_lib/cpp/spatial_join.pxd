# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from cudf._lib.column cimport column_view
from cudf._lib.table cimport table, table_view
from cudf._lib.move cimport move, unique_ptr

cdef extern from "cuspatial/spatial_join.hpp" namespace "cuspatial" nogil:

    cdef unique_ptr[table] join_quadtree_and_bounding_boxes \
        "cuspatial::join_quadtree_and_bounding_boxes" (
        const table_view & quadtree,
        const table_view & poly_bbox,
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

    cdef unique_ptr[table] quadtree_point_to_nearest_polyline \
        "cuspatial::quadtree_point_to_nearest_polyline" (
        const table_view & poly_quad_pairs,
        const table_view & quadtree,
        const column_view & point_indices,
        const column_view & points_x,
        const column_view & points_y,
        const column_view & poly_offsets,
        const column_view & poly_points_x,
        const column_view & poly_points_y
    ) except +
