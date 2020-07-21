# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.column cimport column, column_view
from cudf._lib.move cimport unique_ptr

cdef extern from "cuspatial/polygon_distance.hpp" namespace "cuspatial" \
        nogil:
    cdef unique_ptr[column] directed_polygon_distance(
        const column_view & xs,
        const column_view & ys,
        const column_view & space_offsets
    ) except +
