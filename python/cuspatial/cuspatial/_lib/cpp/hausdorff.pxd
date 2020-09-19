# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

from libcpp.memory cimport unique_ptr


cdef extern from "cuspatial/hausdorff.hpp" namespace "cuspatial" nogil:

    cdef unique_ptr[column] directed_hausdorff_distance(
        const column_view& xs,
        const column_view& ys,
        const column_view& points_per_space
    ) except +
