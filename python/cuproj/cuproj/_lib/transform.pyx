import cupy as cp

from libc.stdint cimport uintptr_t

from libcpp.memory cimport unique_ptr

from cuproj._lib.cpp.projection cimport projection
from cuproj._lib.cpp.operation cimport direction

from cuproj._lib.cpp.cuprojshim cimport vec_2d, make_projection, transform


# Assumption: srcarr is a (N, 2) shaped cupy array
def wgs84_to_utm(srcarr):
    # allocate C-contiguous array
    result = cp.array(srcarr.shape, order='C', dtype=cp.float64)

    cdef unique_ptr[projection[vec_2d[double]]] proj = \
        make_projection(b"epsg:4326", b"epsg:32633")

    c_srcarr = cp.asccontiguousarray(srcarr)

    cdef size_t num_coordinates = srcarr.shape[0]

    cdef vec_2d[double]* input_coords_begin = <vec_2d[double]*> <uintptr_t> c_srcarr.data.ptr
    cdef vec_2d[double]* output_coords = <vec_2d[double]*> <uintptr_t> result.data.ptr

    cdef direction d = direction.FORWARD

    with nogil:
        transform(
            proj.get()[0],
            input_coords_begin,
            output_coords,
            num_coordinates,
            d
        )

    return result
