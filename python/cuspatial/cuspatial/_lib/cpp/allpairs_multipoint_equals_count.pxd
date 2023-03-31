# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view


cdef extern from "cuspatial/allpairs_point_in_polygon.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] allpairs_point_in_polygon(
        const column_view & lhs,
        const column_view & rhs,
    ) except +
