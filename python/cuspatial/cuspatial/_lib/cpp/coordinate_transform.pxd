# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr


cdef extern from "cuspatial/coordinate_transform.hpp" namespace "cuspatial" \
        nogil:
    cdef pair[unique_ptr[column], unique_ptr[column]] lonlat_to_cartesian(
        const double origin_lon,
        const double origin_lat,
        const column_view& input_lon,
        const column_view& input_lat
    ) except +
