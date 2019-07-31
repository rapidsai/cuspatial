from cuspatial.bindings.cudf_cpp import *
from cudf.dataframe.column import Column

from libc.stdlib cimport calloc, malloc, free
                        
cpdef cpp_pip2_bm(pnt_x,pnt_y,ply_fpos,ply_rpos,ply_x,ply_y): 
    print("in cpp_pip2_bm")
    cdef gdf_column* c_pnt_x = column_view_from_column(pnt_x)
    cdef gdf_column* c_pnt_y = column_view_from_column(pnt_y)
    
    cdef gdf_column* c_ply_fpos = column_view_from_column(ply_fpos)
    cdef gdf_column* c_ply_rpos = column_view_from_column(ply_rpos)
 
    cdef gdf_column* c_ply_x = column_view_from_column(ply_x)
    cdef gdf_column* c_ply_y = column_view_from_column(ply_y)    
    cdef gdf_column* res_bm = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        res_bm[0] = pip2_bm(c_pnt_x[0],c_pnt_y[0],c_ply_fpos[0],c_ply_rpos[0],c_ply_x[0],c_ply_y[0])
        
    data, mask = gdf_column_to_column_mem(res_bm)
    free(c_pnt_x)
    free(c_pnt_y)
    free(c_ply_fpos)
    free(c_ply_rpos)
    free(c_ply_x)
    free(c_ply_y)  
    free(res_bm)
    bm=Column.from_mem_views(data, mask) 
    
    return bm                                
