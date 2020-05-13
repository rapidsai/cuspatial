# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.move cimport *


# Note: declaring `move()` with `except +` doesn't work.
#
# Consider:
#     cdef unique_ptr[int] x = move(y)
#
# If `move()` is declared with `except +`, the generated C++ code
# looks something like this:
#
#    std::unique_ptr<int>  __pyx_v_x;
#    std::unique_ptr<int>  __pyx_v_y;
#    std::unique_ptr<int>  __pyx_t_1;
#    try {
#      __pyx_t_1 = std::move(__pyx_v_y);
#    } catch(...) {
#      __Pyx_CppExn2PyErr();
#      __PYX_ERR(0, 8, __pyx_L1_error)
#    }
#    __pyx_v_x = __pyx_t_1;
#
# where the last statement will result in a compiler error
# (copying a unique_ptr).
#
cdef extern from "<utility>" namespace "std" nogil:
    cdef pair[unique_ptr[column], unique_ptr[column]] move(
        pair[unique_ptr[column], unique_ptr[column]]
    )
    cdef pair[unique_ptr[table], unique_ptr[column]] move(
        pair[unique_ptr[table], unique_ptr[column]]
    )
