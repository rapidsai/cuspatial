# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.table cimport *

cdef extern from "cubicspline.hpp" namespace "cuspatial":
    cdef unique_ptr[table] cpp_cubicspline "cuspatial::cubicspline" (
        column_view x,
        table_view y,
        table_view ids_and_end_coordinates
    ) except +
    
cdef extern from "cubicspline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_cubicspline_thrust "cuspatial::cubicspline_column" (
        column_view t,
        column_view x,
        column_view ids
    ) except +

cdef extern from "cubicspline.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] cpp_cubicspline_cusparse "cuspatial::cubicspline_full" (
        column_view t,
        column_view x,
        column_view ids,
        column_view prefix_sums
    ) except +
