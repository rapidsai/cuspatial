# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.column cimport column, column_view
from cudf._lib.table cimport table, table_view

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr

cdef extern from "cuspatial/spatial_window.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] points_in_spatial_window \
        "cuspatial::points_in_spatial_window" (
        double window_min_x,
        double window_max_x,
        double window_min_y,
        double window_max_y,
        const column_view & x,
        const column_view & y
    ) except +
