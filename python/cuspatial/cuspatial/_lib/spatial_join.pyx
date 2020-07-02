# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport size_type
from cudf._lib.table cimport table, table_view, Table

from cuspatial._lib.cpp.spatial_join cimport (
    quad_bbox_join as cpp_quad_bbox_join,
)

from cuspatial._lib.move cimport move

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

cpdef quad_bbox_join(Table quadtree,
                     Table poly_bounding_boxes,
                     double x_min,
                     double x_max,
                     double y_min,
                     double y_max,
                     double scale,
                     size_type max_depth):
    cdef table_view c_quadtree = quadtree.data_view()
    cdef table_view c_poly_bounding_boxes = poly_bounding_boxes.data_view()
    cdef unique_ptr[table] result
    with nogil:
        result = move(cpp_quad_bbox_join(
            c_quadtree,
            c_poly_bounding_boxes,
            x_min, x_max, y_min, y_max, scale, max_depth
        ))
    return Table.from_unique_ptr(
        move(result),
        column_names=["poly_offset", "quad_offset"]
    )
