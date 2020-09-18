# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport size_type
from cudf._lib.table cimport table, table_view, Table
from cudf._lib.column cimport column, column_view, Column

from cuspatial._lib.cpp.spatial_join cimport (
    join_quadtree_and_bounding_boxes as cpp_join_quadtree_and_bounding_boxes,
    quadtree_point_in_polygon as cpp_quadtree_pip,
    quadtree_point_to_nearest_polyline as cpp_quadtree_p2p,
)

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libc.stdint cimport int8_t
from libcpp.utility cimport move

cpdef join_quadtree_and_bounding_boxes(Table quadtree,
                                       Table poly_bounding_boxes,
                                       double x_min,
                                       double x_max,
                                       double y_min,
                                       double y_max,
                                       double scale,
                                       int8_t max_depth):
    cdef table_view c_quadtree = quadtree.data_view()
    cdef table_view c_poly_bounding_boxes = poly_bounding_boxes.data_view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_join_quadtree_and_bounding_boxes(
            c_quadtree,
            c_poly_bounding_boxes,
            x_min, x_max, y_min, y_max, scale, max_depth
        ))
    return Table.from_unique_ptr(
        move(result),
        column_names=["poly_offset", "quad_offset"]
    )


cpdef quadtree_point_in_polygon(Table poly_quad_pairs,
                                Table quadtree,
                                Column point_indices,
                                Column points_x,
                                Column points_y,
                                Column poly_offsets,
                                Column ring_offsets,
                                Column poly_points_x,
                                Column poly_points_y):
    cdef table_view c_poly_quad_pairs = poly_quad_pairs.data_view()
    cdef table_view c_quadtree = quadtree.data_view()
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
    return Table.from_unique_ptr(
        move(result),
        column_names=["polygon_index", "point_index"]
    )


cpdef quadtree_point_to_nearest_polyline(Table poly_quad_pairs,
                                         Table quadtree,
                                         Column point_indices,
                                         Column points_x,
                                         Column points_y,
                                         Column poly_offsets,
                                         Column poly_points_x,
                                         Column poly_points_y):
    cdef table_view c_poly_quad_pairs = poly_quad_pairs.data_view()
    cdef table_view c_quadtree = quadtree.data_view()
    cdef column_view c_point_indices = point_indices.view()
    cdef column_view c_points_x = points_x.view()
    cdef column_view c_points_y = points_y.view()
    cdef column_view c_poly_offsets = poly_offsets.view()
    cdef column_view c_poly_points_x = poly_points_x.view()
    cdef column_view c_poly_points_y = poly_points_y.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_quadtree_p2p(
            c_poly_quad_pairs,
            c_quadtree,
            c_point_indices,
            c_points_x,
            c_points_y,
            c_poly_offsets,
            c_poly_points_x,
            c_poly_points_y
        ))
    return Table.from_unique_ptr(
        move(result),
        column_names=["point_index", "polyline_index", "distance"]
    )
