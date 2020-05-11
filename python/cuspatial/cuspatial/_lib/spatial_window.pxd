# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.legacy.cudf cimport *

from cudf._lib.column cimport column, column_view
from cudf._lib.table cimport table, table_view
from cudf._lib.move cimport move, unique_ptr

cdef extern from "spatial_window.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_points_in_spatial_window \
        "cuspatial::points_in_spatial_window" (
        double window_min_x,
        double window_max_x,
        double window_min_y,
        double window_max_y,
        const column_view & x,
        const column_view & y
    ) except +
