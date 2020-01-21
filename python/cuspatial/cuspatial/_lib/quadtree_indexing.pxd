# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.lib cimport unique_ptr,table,column,column_view

cdef extern from "cuspatial/quadtree.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] quadtree_on_points(
        column_view x,column_view y
    ) except +

cdef extern from "cuspatial/quadtree.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[column] nested_column_test(
        column_view x,column_view y
    ) except +
