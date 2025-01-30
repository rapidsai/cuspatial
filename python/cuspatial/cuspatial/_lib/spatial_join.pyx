# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from pylibcudf cimport Column as plc_Column, Table as plc_Table
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table, table_view

from cuspatial._lib.cpp.spatial_join cimport (
    join_quadtree_and_bounding_boxes as cpp_join_quadtree_and_bounding_boxes,
    quadtree_point_in_polygon as cpp_quadtree_pip,
    quadtree_point_to_nearest_linestring as cpp_quadtree_p2p,
)


cpdef plc_Table join_quadtree_and_bounding_boxes(
    object quadtree,
    object bounding_boxes,
    double x_min,
    double x_max,
    double y_min,
    double y_max,
    double scale,
    int8_t max_depth
):
    cdef plc_Table plc_quadtree = plc_Table(
        [col.to_pylibcudf(mode="read") for col in quadtree._columns]
    )
    cdef table_view c_quadtree = plc_quadtree.view()
    cdef plc_Table plc_bounding_boxes = plc_Table(
        [col.to_pylibcudf(mode="read") for col in bounding_boxes._columns]
    )
    cdef table_view c_bounding_boxes = plc_bounding_boxes.view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_join_quadtree_and_bounding_boxes(
            c_quadtree,
            c_bounding_boxes,
            x_min, x_max, y_min, y_max, scale, max_depth
        ))
    return plc_Table.from_libcudf(move(result))


cpdef plc_Table quadtree_point_in_polygon(
    object poly_quad_pairs,
    object quadtree,
    plc_Column point_indices,
    plc_Column points_x,
    plc_Column points_y,
    plc_Column poly_offsets,
    plc_Column ring_offsets,
    plc_Column poly_points_x,
    plc_Column poly_points_y
):
    cdef plc_Table plc_poly_quad_pairs = plc_Table(
        [col.to_pylibcudf(mode="read") for col in poly_quad_pairs._columns]
    )
    cdef table_view c_poly_quad_pairs = plc_poly_quad_pairs.view()
    cdef plc_Table plc_quadtree = plc_Table(
        [col.to_pylibcudf(mode="read") for col in quadtree._columns]
    )
    cdef table_view c_quadtree = plc_quadtree.view()
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
    return plc_Table.from_libcudf(move(result))


cpdef plc_Table quadtree_point_to_nearest_linestring(
    object linestring_quad_pairs,
    object quadtree,
    plc_Column point_indices,
    plc_Column points_x,
    plc_Column points_y,
    plc_Column linestring_offsets,
    plc_Column linestring_points_x,
    plc_Column linestring_points_y
):
    cdef plc_Table plc_quad_pairs = plc_Table(
        [
            col.to_pylibcudf(mode="read")
            for col in linestring_quad_pairs._columns
        ]
    )
    cdef table_view c_linestring_quad_pairs = plc_quad_pairs.view()
    cdef plc_Table plc_quadtree = plc_Table(
        [col.to_pylibcudf(mode="read") for col in quadtree._columns]
    )
    cdef table_view c_quadtree = plc_quadtree.view()
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
    return plc_Table.from_libcudf(move(result))
