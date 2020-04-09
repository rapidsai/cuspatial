# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from cudf._lib.cpp.column.column cimport column
from cudf._lib.move cimport *

cdef extern from "<utility>" namespace "std" nogil:
    cdef pair[unique_ptr[column], unique_ptr[column]] move(
        pair[unique_ptr[column], unique_ptr[column]]
    )
