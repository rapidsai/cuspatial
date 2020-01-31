# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport make_unique
from cudf._libxx.table cimport *

cpdef cubicspline(Column x, Table y, Table ids):
    cdef unique_ptr[table] c_result = move(cpp_cubicspline(x.view(), y.view(), ids.view()))
    return Table.from_unique_ptr(move(c_result), ['a', 'b', 'c', 'd'])

cpdef cubicspline_column(Column t, Column x, Column ids):
    t_v = t.view()
    x_v = x.view()
    ids_v = ids.view()
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(cpp_cubicspline_column(t_v, x_v, ids_v))
    result = Table.from_unique_ptr(move(c_result), [1,2,3,4])
    return result
