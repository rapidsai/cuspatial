# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.column cimport Column, column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport data_from_unique_ptr, table_view_from_table

from cuspatial._lib.cpp.interpolate cimport (
    cubicspline_coefficients as cpp_cubicspline_coefficients,
    cubicspline_interpolate as cpp_cubicspline_interpolate,
)


cpdef cubicspline_coefficients(
    Column t,
    Column x,
    Column ids,
    Column prefixes
):
    t_v = t.view()
    x_v = x.view()
    ids_v = ids.view()
    prefixes_v = prefixes.view()
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(
            cpp_cubicspline_coefficients(
                t_v,
                x_v,
                ids_v,
                prefixes_v
            )
        )
    return data_from_unique_ptr(move(c_result), ["d3", "d2", "d1", "d0"])

cpdef cubicspline_interpolate(
    Column points,
    Column ids,
    Column prefixes,
    Column original_t,
    object coefficients
):
    p_v = points.view()
    ids_v = ids.view()
    prefixes_v = prefixes.view()
    original_t_v = original_t.view()
    coefs_v = table_view_from_table(coefficients, ignore_index=True)
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(
            cpp_cubicspline_interpolate(
                p_v,
                ids_v,
                prefixes_v,
                original_t_v,
                coefs_v
            )
        )
    result = Column.from_unique_ptr(move(c_result))
    return result
