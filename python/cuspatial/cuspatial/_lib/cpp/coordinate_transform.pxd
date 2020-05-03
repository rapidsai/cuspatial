# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cuspatial/coordinate_transform.hpp" namespace "cuspatial" nogil:
    cdef pair[unique_ptr[column], unique_ptr[column]] lonlat_to_cartesian(
        const double origin_lon,
        const double origin_lat,
        const column_view& input_lon,
        const column_view& input_lat
    ) except +
