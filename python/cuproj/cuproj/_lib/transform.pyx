import cupy as cp

from cuproj._lib.cpp.projection cimport projection
from cuproj._lib.cpp.projection_factories cimport make_projection
from cuproj._lib.cpp.operation cimport direction

def wgs84_to_utm(srcarr):
    result = cp.array(srcarr.shape, dtype=cp.float64)
    cdef projection[double] proj = make_projection[double](
        b"epsg:4326", b"epsg:32633"
    )

    srcarr_data = cp.ascontiguousarray(srcarr, dtype=cp.float64)
    result_data = cp.ascontiguousarray(result, dtype=cp.float64)

    cdef double[::1] srcarr_arr_memview = srcarr_data
    cdef int size = srcarr_data.shape[0]
    cdef double[::1] result_arr_memview = result_data

    with nogil:
        proj.transform(
            &srcarr_arr_memview[0],
            &srcarr_arr_memview[0] + size,
            &result_arr_memview[0],
            direction.FORWARD
        )

    return result
