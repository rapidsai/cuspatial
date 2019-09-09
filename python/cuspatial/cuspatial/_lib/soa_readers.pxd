# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cudf._lib.cudf cimport *
from libcpp.pair cimport pair

cdef extern from "soa_readers.hpp" namespace "cuspatial" nogil:
    cdef gdf_column read_uint32_soa(
        const char *pnt_fn
    ) except +

    cdef gdf_column read_timestamp_soa(
        const char *ts_fn
    ) except +

    cdef pair[gdf_column, gdf_column] read_lonlat_points_soa(
        const char *pnt_fn
    ) except +

    cdef pair[gdf_column, gdf_column] read_xy_points_soa(
        const char *pnt_fn
    ) except +

    cdef void read_polygon_soa(
        const char *ply_fn,
        gdf_column* ply_fpos,
        gdf_column* ply_rpos,
        gdf_column* ply_x,
        gdf_column* ply_y
    ) except +
