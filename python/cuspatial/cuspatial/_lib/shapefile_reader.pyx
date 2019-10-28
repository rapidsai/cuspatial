# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf.core.column import Column
from cudf._lib.cudf import *
from libc.stdlib cimport calloc, malloc, free
from libcpp.pair cimport pair

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

    f_data, f_mask = gdf_column_to_column_mem(c_ply_fpos)
    f_pos = Column.from_mem_views(f_data, f_mask)
    r_data, r_mask = gdf_column_to_column_mem(c_ply_rpos)
    r_pos = Column.from_mem_views(r_data, r_mask)
    x_data, x_mask = gdf_column_to_column_mem(c_ply_x)
    x = Column.from_mem_views(x_data, x_mask)
    y_data, y_mask = gdf_column_to_column_mem(c_ply_y)
    y = Column.from_mem_views(y_data, y_mask)

    return f_pos, r_pos, x, y
