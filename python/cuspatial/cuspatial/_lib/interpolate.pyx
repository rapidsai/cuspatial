# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport make_unique
from cudf._libxx.table cimport *

cpdef cpp_cubicspline(Column x, _Table y, _Table ids):
    cdef unique_ptr[table] c_result 
    x_v = x.view()
    y_v = y.view()
    ids_v = ids.view()
    with nogil:
        c_result = cubicspline(x_v, y_v, ids_v)

    result = _Table.from_ptr(move(c_result))
    return result
