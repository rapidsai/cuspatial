# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column, column, column_view

from cuspatial._lib.cpp.allpairs_multipoint_equals_count cimport (
    allpairs_multipoint_equals_count as cpp_allpairs_multipoint_equals_count,
)


def allpairs_multipoint_equals_count(
    Column _lhs,
    Column _rhs,
):
    cdef column_view lhs = _lhs.view()
    cdef column_view rhs = _rhs.view()

    cdef unique_ptr[column] result

    with nogil:
        result = move(
            cpp_allpairs_multipoint_equals_count(
                lhs,
                rhs,
            )
        )

    return Column.from_unique_ptr(move(result))
