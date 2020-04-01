# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf._lib.legacy.cudf import *
from libc.stdlib cimport calloc, malloc, free
from libcpp.pair cimport pair

cpdef cpp_read_uint_soa(soa_file_name):
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column c_id

    with nogil:
        c_id = read_uint32_soa(
            c_string
        )

    id = gdf_column_to_column(&c_id)

    return id

cpdef cpp_read_ts_soa(soa_file_name):
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column c_ts

    with nogil:
        c_ts = read_timestamp_soa(
            c_string
        )

    ts = gdf_column_to_column(&c_ts)

    return ts

cpdef cpp_read_pnt_lonlat_soa(soa_file_name):
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef pair[gdf_column, gdf_column] columns

    with nogil:
        columns = read_lonlat_points_soa(
            c_string
        )

    lon = gdf_column_to_column(&columns.first)
    lat = gdf_column_to_column(&columns.second)

    return lon, lat

cpdef cpp_read_pnt_xy_soa(soa_file_name):
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef pair[gdf_column, gdf_column] columns

    with nogil:
        columns = read_xy_points_soa(
            c_string
        )

    x = gdf_column_to_column(&columns.first)
    y = gdf_column_to_column(&columns.second)

    return x, y

cpdef cpp_read_polygon_soa(soa_file_name):
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column* c_ply_fpos = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_rpos = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_x = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_y = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        read_polygon_soa(
            c_string,
            c_ply_fpos,
            c_ply_rpos,
            c_ply_x,
            c_ply_y
        )

    f_pos = gdf_column_to_column(c_ply_fpos)
    r_pos = gdf_column_to_column(c_ply_rpos)
    x = gdf_column_to_column(c_ply_x)
    y = gdf_column_to_column(c_ply_y)

    return f_pos, r_pos, x, y
