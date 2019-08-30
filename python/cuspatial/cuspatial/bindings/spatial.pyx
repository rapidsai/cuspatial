from cudf.bindings.cudf_cpp import *
from cudf.dataframe.column import Column
from libcpp.pair cimport pair

from libc.stdlib cimport calloc, malloc, free
                        
cpdef cpp_pip_bm(pnt_x,pnt_y,ply_fpos,ply_rpos,ply_x,ply_y): 
    print("in cpp_pip_bm")
    cdef gdf_column* c_pnt_x = column_view_from_column(pnt_x)
    cdef gdf_column* c_pnt_y = column_view_from_column(pnt_y)
    
    cdef gdf_column* c_ply_fpos = column_view_from_column(ply_fpos)
    cdef gdf_column* c_ply_rpos = column_view_from_column(ply_rpos)
 
    cdef gdf_column* c_ply_x = column_view_from_column(ply_x)
    cdef gdf_column* c_ply_y = column_view_from_column(ply_y)    
    cdef gdf_column* res_bm = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        res_bm[0] = pip_bm(c_pnt_x[0],c_pnt_y[0],c_ply_fpos[0],c_ply_rpos[0],c_ply_x[0],c_ply_y[0])
        
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

cpdef cpp_lonlat2coord(cam_lon, cam_lat, in_lon, in_lat):
    # print("in cpp_ll2coord")
    
    cdef gdf_scalar* c_cam_lon = gdf_scalar_from_scalar(cam_lon)
    cdef gdf_scalar* c_cam_lat = gdf_scalar_from_scalar(cam_lat)
   
    cdef gdf_column* c_in_lon = column_view_from_column(in_lon)
    cdef gdf_column* c_in_lat = column_view_from_column(in_lat)

    cpdef pair[gdf_column, gdf_column] coords

    with nogil:
       coords = lonlat_to_coord(c_cam_lon[0], c_cam_lat[0], c_in_lon[0], c_in_lat[0])

    x_data, x_mask = gdf_column_to_column_mem(&coords.first)
    y_data, y_mask = gdf_column_to_column_mem(&coords.second)
    
    free(c_in_lon)
    free(c_in_lat)
    
    x=Column.from_mem_views(x_data, x_mask)
    y=Column.from_mem_views(y_data, y_mask)

    return x,y
 
cpdef cpp_directed_hausdorff_distance(coor_x,coor_y,cnt):
    print("in cpp_hausdorff_distance")
    cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
    cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
    cdef gdf_column* c_cnt = column_view_from_column(cnt)
    cdef gdf_column* c_dist = <gdf_column*>malloc(sizeof(gdf_column))
    with nogil:
     c_dist[0]=directed_hausdorff_distance(c_coor_x[0],c_coor_y[0],c_cnt[0])
    
    dist_data, dist_mask = gdf_column_to_column_mem(c_dist)    
    dist=Column.from_mem_views(dist_data,dist_mask)
    
    return dist   