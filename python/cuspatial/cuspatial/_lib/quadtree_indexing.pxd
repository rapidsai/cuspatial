# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.lib cimport unique_ptr,table_view,column_view

cdef extern from "cudf/copying.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table_view] quadtree_on_points(
        column_view x,column_view y
    ) except +
