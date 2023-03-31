# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.column cimport column, column_view


cdef extern from "cuspatial/allpairs_multipoint_equals_count.hpp" \
        namespace "cuspatial" nogil:
    cdef unique_ptr[column] allpairs_multipoint_equals_count(
        const column_view & lhs,
        const column_view & rhs,
    ) except +
