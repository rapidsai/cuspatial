# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cuspatial/projection.hpp" namespace "cuspatial" \
        nogil:
    cdef pair[unique_ptr[column], unique_ptr[column]] sinusoidal_projection(
        const double origin_lon,
        const double origin_lat,
        const column_view& input_lon,
        const column_view& input_lat
    ) except +
