# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column, column

from cuspatial._lib.cpp.column.geometry_column_view cimport (
    geometry_column_view,
)
from cuspatial._lib.cpp.pairwise_multipoint_equals_count cimport (
    pairwise_multipoint_equals_count as cpp_pairwise_multipoint_equals_count,
)
from cuspatial._lib.cpp.types cimport collection_type_id, geometry_type_id


def pairwise_multipoint_equals_count(
    Column _lhs,
    Column _rhs,
):
    cdef shared_ptr[geometry_column_view] lhs = \
        make_shared[geometry_column_view](
            _lhs.view(),
            collection_type_id.MULTI,
            geometry_type_id.POINT)

    cdef shared_ptr[geometry_column_view] rhs = \
        make_shared[geometry_column_view](
            _rhs.view(),
            collection_type_id.MULTI,
            geometry_type_id.POINT)

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_pairwise_multipoint_equals_count(
                lhs.get()[0],
                rhs.get()[0],
            )
        )

    return Column.from_unique_ptr(move(result))
