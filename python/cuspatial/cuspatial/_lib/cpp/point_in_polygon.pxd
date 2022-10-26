# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

from cudf._lib.column cimport column, column_view
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cuspatial/point_in_polygon.hpp" namespace "cuspatial" nogil:
    cdef pair[unique_ptr[column], table_view] point_in_polygon(
        const column_view & test_points_x,
        const column_view & test_points_y,
        const column_view & poly_offsets,
        const column_view & poly_ring_offsets,
        const column_view & poly_points_x,
        const column_view & poly_points_y
    ) except +
