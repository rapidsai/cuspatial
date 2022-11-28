# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column, column_view
from cudf._lib.cpp.table.table cimport table, table_view
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table

from cuspatial._lib.cpp.spatial_join cimport (
    join_quadtree_and_bounding_boxes as cpp_join_quadtree_and_bounding_boxes,
    quadtree_point_in_polygon as cpp_quadtree_pip,
    quadtree_point_to_nearest_linestring as cpp_quadtree_p2p,
)


cpdef join_quadtree_and_bounding_boxes(object quadtree,
                                       object bounding_boxes,
                                       double x_min,
                                       double x_max,
                                       double y_min,
                                       double y_max,
                                       double scale,
                                       int8_t max_depth):
    cdef table_view c_quadtree = table_view_from_table(
        quadtree, ignore_index=True)
    cdef table_view c_bounding_boxes = table_view_from_table(
        bounding_boxes, ignore_index=True)
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_join_quadtree_and_bounding_boxes(
            c_quadtree,
            c_bounding_boxes,
            x_min, x_max, y_min, y_max, scale, max_depth
        ))
    return data_from_unique_ptr(
        move(result),
        column_names=["bbox_offset", "quad_offset"]
    )


cpdef quadtree_point_in_polygon(object poly_quad_pairs,
                                object quadtree,
                                Column point_indices,
                                Column points_x,
                                Column points_y,
                                Column poly_offsets,
                                Column ring_offsets,
                                Column poly_points_x,
                                Column poly_points_y):
    cdef table_view c_poly_quad_pairs = table_view_from_table(
        poly_quad_pairs, ignore_index=True)
    cdef table_view c_quadtree = table_view_from_table(
        quadtree, ignore_index=True)
    cdef column_view c_point_indices = point_indices.view()
    cdef column_view c_points_x = points_x.view()
    cdef column_view c_points_y = points_y.view()
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_ring_offsets = ring_offsets.view()
    cdef column_view c_poly_points_x = poly_points_x.view()
    cdef column_view c_poly_points_y = poly_points_y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_quadtree_pip(
            c_poly_quad_pairs,
            c_quadtree,
            c_point_indices,
            c_points_x,
            c_points_y,
            c_poly_offsets,
            c_ring_offsets,
            c_poly_points_x,
            c_poly_points_y
        ))
    return data_from_unique_ptr(
        move(result),
        column_names=["polygon_index", "point_index"]
    )


cpdef quadtree_point_to_nearest_linestring(object linestring_quad_pairs,
                                           object quadtree,
                                           Column point_indices,
                                           Column points_x,
                                           Column points_y,
                                           Column linestring_offsets,
                                           Column linestring_points_x,
                                           Column linestring_points_y):
    cdef table_view c_linestring_quad_pairs = table_view_from_table(
        linestring_quad_pairs, ignore_index=True)
    cdef table_view c_quadtree = table_view_from_table(
        quadtree, ignore_index=True)
    cdef column_view c_point_indices = point_indices.view()
    cdef column_view c_points_x = points_x.view()
    cdef column_view c_points_y = points_y.view()
    cdef column_view c_linestring_offsets = linestring_offsets.view()
    cdef column_view c_linestring_points_x = linestring_points_x.view()
    cdef column_view c_linestring_points_y = linestring_points_y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_quadtree_p2p(
            c_linestring_quad_pairs,
            c_quadtree,
            c_point_indices,
            c_points_x,
            c_points_y,
            c_linestring_offsets,
            c_linestring_points_x,
            c_linestring_points_y
        ))
    return data_from_unique_ptr(
        move(result),
        column_names=["point_index", "linestring_index", "distance"]
    )
