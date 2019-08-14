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

cpdef cpp_haversine_distance(x1,y1,x2,y2): 
    print("in cpp_haversine_distance")
    cdef gdf_column* c_x1= column_view_from_column(x1)
    cdef gdf_column* c_y1 = column_view_from_column(y1)
    cdef gdf_column* c_x2= column_view_from_column(x2)
    cdef gdf_column* c_y2 = column_view_from_column(y2)
 
    cdef gdf_column* c_h_dist = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        c_h_dist[0] =haversine_distance(c_x1[0],c_y1[0],c_x2[0],c_y2[0])
        
    data, mask = gdf_column_to_column_mem(c_h_dist)
    free(c_x1)
    free(c_y1)
    free(c_x2)
    free(c_y2)
    free(c_h_dist)
    h_dist=Column.from_mem_views(data, mask) 
    
    return h_dist  

cpdef cpp_ll2coor(cam_x,cam_y,in_x,in_y): 
    print("in cpp_ll2coord")
    
    cdef gdf_scalar* c_cam_x=gdf_scalar_from_scalar(cam_x)
    cdef gdf_scalar* c_cam_y=gdf_scalar_from_scalar(cam_y)
   
    cdef gdf_column* c_in_x = column_view_from_column(in_x)
    cdef gdf_column* c_in_y = column_view_from_column(in_y)

    cdef gdf_column* c_out_x = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_out_y = <gdf_column*>malloc(sizeof(gdf_column))
 
    with nogil:
       ll2coor(c_cam_x[0],c_cam_y[0],c_in_x[0],c_in_y[0],c_out_x[0],c_out_y[0])
        
    x_data, x_mask = gdf_column_to_column_mem(c_out_x)
    y_data, y_mask = gdf_column_to_column_mem(c_out_y)
    
    free(c_in_x)
    free(c_in_y)
    free(c_out_x)
    free(c_out_y)
    
    x=Column.from_mem_views(x_data, x_mask)
    y=Column.from_mem_views(y_data, y_mask)
    
    return x,y         
 
cpdef cpp_directed_hausdorff(coor_x,coor_y,cnt):
    print("in cpp_hausdorff_distance")
    cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
    cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
    cdef gdf_column* c_cnt = column_view_from_column(cnt)
    cdef gdf_column* c_dist = <gdf_column*>malloc(sizeof(gdf_column))
    with nogil:
     c_dist[0]=hausdorff_distance(c_coor_x[0],c_coor_y[0],c_cnt[0])
    
    dist_data, dist_mask = gdf_column_to_column_mem(c_dist)    
    dist=Column.from_mem_views(dist_data,dist_mask)
    
    return dist   