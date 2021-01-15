# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf._lib.column cimport column, column_view, Column

from cuspatial._lib.cpp.hausdorff \
    cimport directed_hausdorff_distance as directed_cpp_hausdorff_distance

from libcpp.utility cimport move
from libcpp.memory cimport unique_ptr


def directed_hausdorff_distance(
    Column xs,
    Column ys,
    Column space_offsets,
):
    cdef column_view c_xs = xs.view()
    cdef column_view c_ys = ys.view()
    cdef column_view c_shape_offsets = space_offsets.view()

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            directed_cpp_hausdorff_distance(
                c_xs,
                c_ys,
                c_shape_offsets,
            )
        )

    return Column.from_unique_ptr(move(result))
