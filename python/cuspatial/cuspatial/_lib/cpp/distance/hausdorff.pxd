# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cuspatial/distance/hausdorff.hpp" \
        namespace "cuspatial" nogil:

    cdef unique_ptr[column] directed_hausdorff_distance(
        const column_view& xs,
        const column_view& ys,
        const column_view& space_offsets
    ) except +
