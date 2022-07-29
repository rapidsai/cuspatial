# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view


cdef extern from "cuspatial/distance/haversine.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] haversine_distance(
        const column_view& a_lon,
        const column_view& a_lat,
        const column_view& b_lon,
        const column_view& b_lat
    ) except +
