# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cuspatial/distance/hausdorff.hpp" \
        namespace "cuspatial" nogil:

    cdef pair[unique_ptr[column], table_view] directed_hausdorff_distance(
        const column_view& xs,
        const column_view& ys,
        const column_view& space_offsets
    ) except +
