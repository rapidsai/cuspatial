# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view
from cudf._lib.cpp.table.table cimport table, table_view


cdef extern from "cuspatial/cubic_spline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cubicspline_coefficients \
        "cuspatial::cubicspline_coefficients" (
        const column_view & t,
        const column_view & x,
        const column_view & ids,
        const column_view & prefix_sums
    ) except +

    cdef unique_ptr[column] cubicspline_interpolate \
        "cuspatial::cubicspline_interpolate" (
        const column_view & p,
        const column_view & ids,
        const column_view & prefix_sums,
        const column_view & old_t,
        const table_view & coefficients
    ) except +
