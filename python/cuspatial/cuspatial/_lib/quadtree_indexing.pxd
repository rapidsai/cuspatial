# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._libxx.lib cimport unique_ptr,table,column,column_view,mutable_column_view

cdef extern from "cuspatial/quadtree.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[table] quadtree_on_points(
        mutable_column_view x,mutable_column_view y,double x1,double y1,double x2,double y2,double scale,int M, int MINSIZE 
    ) except +

cdef extern from "cuspatial/quadtree.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[column] nested_column_test(
        column_view x,column_view y
    ) except +
