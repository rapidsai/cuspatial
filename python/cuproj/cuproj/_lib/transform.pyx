import cupy as cp

from libc.stdint cimport uintptr_t

from cuproj._lib.cpp.cuprojshim cimport make_projection, transform, vec_2d
from cuproj._lib.cpp.operation cimport direction
from cuproj._lib.cpp.projection cimport projection


cdef direction_string_to_enum(dir: str):
    return direction.FORWARD if dir == "FORWARD" else direction.INVERSE

cdef class Transformer:
    cdef projection[vec_2d[double]]* proj

    def __init__(self, crs_from, crs_to):
        self.proj = make_projection(crs_from.encode('utf-8'), crs_to.encode('utf-8'))

    def __del__(self):
        del self.proj


    def transform(self, x, y, dir):
        # Assumption: srcarr is a (N,) shaped cupy array
        cdef int size = x.shape[0]

        # allocate C-contiguous array
        result_x = cp.array((size,), order='C', dtype=cp.float64)
        result_y = cp.array((size,), order='C', dtype=cp.float64)

        cdef double* x_in = <double*> <uintptr_t> x.data.ptr
        cdef double* y_in = <double*> <uintptr_t> y.data.ptr
        cdef double* x_out = <double*> <uintptr_t> result_x.data.ptr
        cdef double* y_out = <double*> <uintptr_t> result_y.data.ptr

        cdef direction d = direction_string_to_enum(dir)

        with nogil:
            transform(
                self.proj[0],
                x_in,
                y_in,
                x_out,
                y_out,
                size,
                d
            )

        return result_x, result_y
