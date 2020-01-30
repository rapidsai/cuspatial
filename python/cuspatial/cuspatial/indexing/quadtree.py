# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf._libxx.column import Column
from cuspatial._lib.quadtree_indexing import cpp_quadtree_on_points
from cuspatial._lib.quadtree_indexing import cpp_nested_column_test


def quadtree_on_points(x, y,x1,y1,x2,y2,scale,M,MINSIZE):
    result = cpp_quadtree_on_points(x,y,x1,y1,x2,y2,scale,M,MINSIZE)
    return result
  
def nested_column_test(x, y):
    result = cpp_nested_column_test(x,y)
    return result