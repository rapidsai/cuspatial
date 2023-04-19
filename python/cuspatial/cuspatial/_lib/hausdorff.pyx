# Copyright (c) 2019, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from cudf._lib.column cimport Column, column, column_view
from cudf._lib.cpp.table.table_view cimport table_view
from cudf._lib.utils cimport columns_from_table_view

from cuspatial._lib.cpp.distance.hausdorff cimport (
    directed_hausdorff_distance as directed_cpp_hausdorff_distance,
)


def directed_hausdorff_distance(
    Column xs,
    Column ys,
    Column space_offsets,
):
    cdef column_view c_xs = xs.view()
    cdef column_view c_ys = ys.view()
    cdef column_view c_shape_offsets = space_offsets.view()

    cdef pair[unique_ptr[column], table_view] result

    with nogil:
        result = move(
            directed_cpp_hausdorff_distance(
                c_xs,
                c_ys,
                c_shape_offsets,
            )
        )

    owner = Column.from_unique_ptr(move(result.first), data_ptr_exposed=True)

    return columns_from_table_view(
        result.second,
        owners=[owner] * result.second.num_columns()
    )
