import cupy as cp

from libc.stdint cimport uintptr_t

from cuproj._lib.cpp.projection cimport projection
from cuproj._lib.cpp.projection_factories cimport make_projection
from cuproj._lib.cpp.operation cimport direction

from cuproj._lib.cpp.cuprojshim cimport vec_2d, transform

# Assumption: srcarr is a (N, 2) shaped cupy array
def wgs84_to_utm(srcarr):
    # allocate C-contiguous array
    result = cp.array(srcarr.shape, order='C', dtype=cp.float64)

    cdef projection[double] proj = make_projection[double](
        b"epsg:4326", b"epsg:32633"
    )

    c_srcarr = cp.asccontiguousarray(srcarr)

    cdef size_t num_coordinates = srcarr.shape[0]

    cdef vec_2d* input_coords_begin = <vec_2d*> <uintptr_t> c_srcarr.data.ptr
    cdef vec_2d* output_coords = <vec_2d*> <uintptr_t> result.data.ptr

    with nogil:
        transform(
            proj,
            input_coords_begin,
            output_coords,
            num_coordinates,
            direction.FORWARD
        )

    return result
