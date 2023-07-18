import cupy as cp

from libc.stdint cimport uintptr_t

from libcpp.memory cimport unique_ptr

from cuproj._lib.cpp.projection cimport projection
from cuproj._lib.cpp.operation cimport direction

from cuproj._lib.cpp.cuprojshim cimport vec_2d, make_projection, transform

cdef direction_string_to_enum(dir: str):
    return direction.FORWARD if dir == "FORWARD" else direction.INVERSE

# Assumption: srcarr is a (N,) shaped cupy array
def wgs84_to_utm(x, y, dir):
    cdef int size = x.shape[0]
    # allocate C-contiguous array
    result_x = cp.array((size,), order='C', dtype=cp.float64)
    result_y = cp.array((size,), order='C', dtype=cp.float64)

    c_x = cp.ascontiguousarray(x)
    c_y = cp.ascontiguousarray(y)

    cdef unique_ptr[projection[vec_2d[double]]] proj = \
        make_projection(b"epsg:4326", b"epsg:32633")
    cdef double* x_in = <double*> <uintptr_t> x.data.ptr
    cdef double* y_in = <double*> <uintptr_t> y.data.ptr
    cdef double* x_out = <double*> <uintptr_t> result_x.data.ptr
    cdef double* y_out = <double*> <uintptr_t> result_y.data.ptr

    cdef direction d = direction_string_to_enum(dir)

    with nogil:
        transform(
            proj.get()[0],
            x_in,
            y_in,
            x_out,
            y_out,
            size,
            d
        )

    return result_x, result_y
