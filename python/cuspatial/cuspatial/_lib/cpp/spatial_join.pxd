# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport size_type
from cudf._lib.column cimport column, column_view
from cudf._lib.table cimport table, table_view
from cudf._lib.move cimport move, unique_ptr

cdef extern from "cuspatial/spatial_join.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] quad_bbox_join \
        "cuspatial::quad_bbox_join" (
        const table_view & quadtree,
        const table_view & poly_bbox,
        double x_min,
        double x_max,
        double y_min,
        double y_max,
        double scale,
        size_type max_depth
    ) except +
