# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table


cdef extern from "cuspatial/point_quadtree.hpp" namespace "cuspatial" nogil:
    cdef pair[unique_ptr[column], unique_ptr[table]] quadtree_on_points(
        column_view x,
        column_view y,
        double x1,
        double y1,
        double x2,
        double y2,
        double scale,
        int num_levels,
        int min_size
    ) except +
