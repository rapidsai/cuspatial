# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.column cimport column, column_view
from cudf._libxx.table cimport table, table_view
from cudf._libxx.move cimport move, unique_ptr

cdef extern from "cubic_spline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_cubicspline_coefficients \
        "cuspatial::cubicspline_coefficients_default" (
        const column_view & t,
        const column_view & x,
        const column_view & ids,
        const column_view & prefix_sums
    ) except +

    cdef unique_ptr[column] cpp_cubicspline_interpolate \
        "cuspatial::cubicspline_interpolate_default" (
        const column_view & p,
        const column_view & ids,
        const column_view & prefix_sums,
        const column_view & old_t,
        const table_view & coefficients
    ) except +
