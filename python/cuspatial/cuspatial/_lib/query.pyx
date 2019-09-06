from cudf.dataframe.column import Column
from cudf._lib.cudf import *

from libc.stdlib cimport calloc, malloc, free

cpdef cpp_spatial_window_points(left, bottom, right, top, x, y): 
    cdef gdf_scalar* c_left = gdf_scalar_from_scalar(left)
    cdef gdf_scalar* c_bottom = gdf_scalar_from_scalar(bottom)
    cdef gdf_scalar* c_right = gdf_scalar_from_scalar(right)
    cdef gdf_scalar* c_top = gdf_scalar_from_scalar(top)
   
    cdef gdf_column* c_x = column_view_from_column(x)
    cdef gdf_column* c_y = column_view_from_column(y)

    cdef gdf_column* c_out_x = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_out_y = <gdf_column*>malloc(sizeof(gdf_column))

    cdef pair[gdf_column, gdf_column] xy

    with nogil:
        xy = spatial_window_points(c_left[0], c_bottom[0], c_right[0], c_top[0],
                                   c_x[0], c_y[0])

    outx_data, outx_mask = gdf_column_to_column_mem(&xy.first)
    outy_data, outy_mask = gdf_column_to_column_mem(&xy.second)

    outx=Column.from_mem_views(outx_data, outx_mask)
    outy=Column.from_mem_views(outy_data, outy_mask)

    return num_hits, outx, outy