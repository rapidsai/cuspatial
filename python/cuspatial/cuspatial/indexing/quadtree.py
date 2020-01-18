# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf._libxx.column import Column
from cuspatial._lib.quadtree_indexing import cpp_quadtree_on_points


def quadtree_on_points(x, y):
    result = cpp_quadtree_on_points(x,y)
    return result
  
