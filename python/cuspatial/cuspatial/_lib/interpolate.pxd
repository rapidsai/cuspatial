# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._lib.column cimport column, column_view
from cudf._lib.table cimport table, table_view

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr

cdef extern from "cuspatial/cubic_spline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_cubicspline_coefficients \
        "cuspatial::cubicspline_coefficients" (
        const column_view & t,
        const column_view & x,
        const column_view & ids,
        const column_view & prefix_sums
    ) except +

    cdef unique_ptr[column] cpp_cubicspline_interpolate \
        "cuspatial::cubicspline_interpolate" (
        const column_view & p,
        const column_view & ids,
        const column_view & prefix_sums,
        const column_view & old_t,
        const table_view & coefficients
    ) except +
