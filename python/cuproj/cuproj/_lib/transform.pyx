import cupy as cp

from libc.stdint cimport uintptr_t
from libcpp.string cimport string

from cuproj._lib.cpp.cuprojshim cimport make_projection, transform, vec_2d
from cuproj._lib.cpp.operation cimport direction
from cuproj._lib.cpp.projection cimport projection


cdef direction_string_to_enum(dir: str):
    return direction.FORWARD if dir == "FORWARD" else direction.INVERSE

cdef class Transformer:
    cdef projection[vec_2d[float]]* proj_32
    cdef projection[vec_2d[double]]* proj_64

    def __init__(self, crs_from, crs_to):
        if isinstance(crs_from, int):
            crs_from = str(crs_from)
        elif isinstance(crs_from, tuple):
            crs_from = str.join(":", crs_from)

        if isinstance(crs_to, int):
            crs_to = str(crs_to)
        elif isinstance(crs_to, tuple):
            crs_to = str.join(":", crs_to)

        if (not isinstance(crs_from, str) or not isinstance(crs_to, str)):
            raise TypeError(
                "crs_from and crs_to must be strings or integers")

        crs_from_b = crs_from.encode('utf-8')
        crs_to_b = crs_to.encode('utf-8')
        self.proj_32 = make_projection[float](
            <string> crs_from_b, <string> crs_to_b)
        self.proj_64 = make_projection[double](
            <string> crs_from_b, <string> crs_to_b)

    def __del__(self):
        del self.proj_32
        del self.proj_64

    def transform(self, x, y, dir):
        if (len(x.shape) != 1):
            raise TypeError("x must be a 1D array")
        if (len(y.shape) != 1):
            raise TypeError("y must be a 1D array")
        if (x.shape[0] != y.shape[0]):
            raise TypeError("x and y must have the same length")
        if isinstance(x.dtype, cp.floating):
            raise TypeError("x must be of floating point type")
        if isinstance(y.dtype, cp.floating):
            raise TypeError("y must be of floating point type")
        if (x.dtype != y.dtype):
            raise TypeError("x and y must have the same dtype")

        if (x.dtype == cp.float32):
            return self.transform_32(x, y, dir)
        else:
            return self.transform_64(x, y, dir)

    def transform_32(self, x, y, dir):
        cdef int size = x.shape[0]
        result_x = cp.ndarray((size,), order='C', dtype=x.dtype)
        result_y = cp.ndarray((size,), order='C', dtype=y.dtype)
        cdef float* x_in = \
            <float*> <uintptr_t> x.__cuda_array_interface__['data'][0]
        cdef float* y_in = \
            <float*> <uintptr_t> y.__cuda_array_interface__['data'][0]
        cdef float* x_out = <float*> <uintptr_t> result_x.data.ptr
        cdef float* y_out = <float*> <uintptr_t> result_y.data.ptr

        cdef direction d = direction_string_to_enum(dir)

        with nogil:
            transform(
                self.proj_32[0],
                x_in,
                y_in,
                x_out,
                y_out,
                size,
                d)

        return result_x, result_y

    def transform_64(self, x, y, dir):
        cdef int size = x.shape[0]
        result_x = cp.ndarray((size,), order='C', dtype=cp.float64)
        result_y = cp.ndarray((size,), order='C', dtype=cp.float64)
        cdef double* x_in = \
            <double*> <uintptr_t> x.__cuda_array_interface__['data'][0]
        cdef double* y_in = \
            <double*> <uintptr_t> y.__cuda_array_interface__['data'][0]
        cdef double* x_out = <double*> <uintptr_t> result_x.data.ptr
        cdef double* y_out = <double*> <uintptr_t> result_y.data.ptr

        cdef direction d = direction_string_to_enum(dir)

        with nogil:
            transform(
                self.proj_64[0],
                x_in,
                y_in,
                x_out,
                y_out,
                size,
                d)

        return result_x, result_y
