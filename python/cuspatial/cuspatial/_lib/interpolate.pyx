# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport make_unique
from cudf._libxx.table cimport *

cpdef cpp_cubicspline(Column x, Table y, Table ids):
    x_v = x.view()
    y_v = y.view()
    ids_v = ids.view()
    cdef unique_ptr[table] c_result = move(cubicspline(x_v, y_v, ids_v))
    result = Table.from_unique_ptr(c_result, ["x", "y"])
    return result

cpdef cpp_cubicspline_column(Column t, Column x, Column ids):
    t_v = t.view()
    x_v = x.view()
    ids_v = ids.view()
    cdef unique_ptr[table] c_result = move(cubicspline_column(t_v, x_v, ids_v))
    result = Table.from_unique_ptr(c_result, ["d3", "d2", "d1", "d0"])
    return move(result)
