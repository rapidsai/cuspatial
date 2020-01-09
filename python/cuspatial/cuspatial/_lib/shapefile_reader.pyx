# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf._lib.cudf import *
from libc.stdlib cimport malloc, free

cpdef cpp_read_polygon_shapefile(shapefile_file_name):
    cdef bytes py_bytes = shapefile_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column* c_ply_fpos = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_rpos = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_x = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_y = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        read_polygon_shapefile(
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

    free(c_ply_fpos)
    free(c_ply_rpos)
    free(c_ply_x)
    free(c_ply_y)

    return f_pos, r_pos, x, y
