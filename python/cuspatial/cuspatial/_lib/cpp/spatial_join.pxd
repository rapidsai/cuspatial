# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
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
        int8_t max_depth
    ) except +
