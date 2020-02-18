# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.table cimport *

cdef extern from "cubicspline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_cubicspline_cusparse "cuspatial::cubicspline_full" (
        column_view t,
        column_view x,
        column_view ids,
        column_view prefix_sums
    ) except +
