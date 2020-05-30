# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf._lib.column cimport column, column_view, Column
from cudf._lib.move cimport move, unique_ptr

from cuspatial._lib.cpp.hausdorff \
    cimport directed_hausdorff_distance as directed_cpp_hausdorff_distance


def directed_hausdorff_distance(
    Column xs,
    Column ys,
    Column points_per_space,
):
    cdef column_view c_xs = xs.view()
    cdef column_view c_ys = ys.view()
    cdef column_view c_points_per_space = points_per_space.view()

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            directed_cpp_hausdorff_distance(
                c_xs,
                c_ys,
                c_points_per_space,
            )
        )

    return Column.from_unique_ptr(move(result))
