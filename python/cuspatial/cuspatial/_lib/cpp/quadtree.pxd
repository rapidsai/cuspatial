# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int8_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cuspatial/point_quadtree.hpp" namespace "cuspatial" nogil:
    cdef pair[unique_ptr[column], unique_ptr[table]] quadtree_on_points(
        column_view x,
        column_view y,
        double x_min,
        double x_max,
        double y_min,
        double y_max,
        double scale,
        int8_t max_depth,
        size_type min_size
    ) except +
