# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.table cimport table, \
                               column_view, \
                               table_view, \
                               move, \
                               unique_ptr, \
                               column

cdef extern from "cubic_spline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_cubicspline_coefficients \
            "cuspatial::cubicspline_coefficients" (
        column_view t,
        column_view x,
        column_view ids,
        column_view prefix_sums
    ) except +

    cdef unique_ptr[column] cpp_cubicspline_interpolate \
            "cuspatial::cubicspline_interpolate" (
        column_view p,
        column_view ids,
        column_view prefix_sums,
        column_view old_t,
        table_view coefficients
    ) except +
