from cudf.core.column import Column
from cudf.core._lib import *

from libc.stdlib cimport calloc, malloc, free

cpdef cpp_sw_xy(qx1,qx2,qy1,qy2,in_x,in_y):
    print("in cpp_sw_xy")

    cdef gdf_scalar* c_qx1=gdf_scalar_from_scalar(qx1)
    cdef gdf_scalar* c_qx2=gdf_scalar_from_scalar(qx2)
    cdef gdf_scalar* c_qy1=gdf_scalar_from_scalar(qy1)
    cdef gdf_scalar* c_qy2=gdf_scalar_from_scalar(qy2)

    cdef gdf_column* c_in_x = column_view_from_column(in_x)
    cdef gdf_column* c_in_y = column_view_from_column(in_y)

    cdef pair[gdf_column, gdf_column] columns

    columns = spatial_window_points(c_qx1, c_qy1, c_qx2, c_qy2,
                                    c_in_x, c_in_y)

    outx_data, outx_mask = gdf_column_to_column_mem(&columns.first)
    outy_data, outy_mask = gdf_column_to_column_mem(&columns.second)

    outx=Column.from_mem_views(outx_data, outx_mask)
    outy=Column.from_mem_views(outy_data, outy_mask)

    return num_hits, outx, outy
