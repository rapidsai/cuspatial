# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view


cdef extern from "cuspatial/point_in_polygon_one_to_one.hpp" namespace "cuspatial" nogil:
    cdef unique_ptr[column] point_in_polygon_one_to_one(
        const column_view & test_points_x,
        const column_view & test_points_y,
        const column_view & poly_offsets,
        const column_view & poly_ring_offsets,
        const column_view & poly_points_x,
        const column_view & poly_points_y
    ) except +
