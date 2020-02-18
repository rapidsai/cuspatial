# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport make_unique
from cudf._libxx.table cimport *

cpdef cubicspline_full(Column t, Column x, Column ids, Column prefixes):
    t_v = t.view()
    x_v = x.view()
    ids_v = ids.view()
    prefixes_v = prefixes.view()
    cdef unique_ptr[table] c_result
    with nogil:
        c_result = move(cpp_cubicspline_cusparse(t_v, x_v, ids_v, prefixes_v))
    result = Table.from_unique_ptr(move(c_result), ["d3", "d2", "d1", "d0"])
    return result
