from cudf.dataframe.column import Column
from cuspatial.bindings.cudf_cpp import *

from libc.stdlib cimport calloc, malloc, free
                        
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
 
cpdef cpp_coor2traj(coor_x,coor_y,pid,ts): 
    print("in cpp_coor2traj")
    cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
    cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
    cdef gdf_column* c_pid = column_view_from_column(pid)
    cdef gdf_column* c_ts = column_view_from_column(ts)
    cdef gdf_column* c_tid = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_len = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_pos = <gdf_column*>malloc(sizeof(gdf_column))
    
    with nogil:
         num_traj=coor2traj(c_coor_x[0],c_coor_y[0],c_pid[0],c_ts[0],c_tid[0],c_len[0],c_pos[0])

    tid_data, tid_mask = gdf_column_to_column_mem(c_tid)   
    len_data, len_mask = gdf_column_to_column_mem(c_len)    
    pos_data, pos_mask = gdf_column_to_column_mem(c_pos)
    tid=Column.from_mem_views(tid_data, tid_mask)
    len=Column.from_mem_views(len_data, len_mask)
    pos=Column.from_mem_views(pos_data, pos_mask)
    
    return num_traj,tid,len,pos

cpdef cpp_traj_distspeed(coor_x,coor_y,ts,len,pos): 
    print("in cpp_coor2traj")
    cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
    cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
    cdef gdf_column* c_ts = column_view_from_column(ts)
    cdef gdf_column* c_len = column_view_from_column(len)
    cdef gdf_column* c_pos = column_view_from_column(pos)
    cdef gdf_column* c_dist = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_speed = <gdf_column*>malloc(sizeof(gdf_column))
    
    with nogil:
         traj_distspeed(c_coor_x[0],c_coor_y[0],c_ts[0],c_len[0],c_pos[0],c_dist[0],c_speed[0])

    dist_data, dist_mask = gdf_column_to_column_mem(c_dist)    
    speed_data, speed_mask = gdf_column_to_column_mem(c_speed)
    dist=Column.from_mem_views(dist_data, dist_mask)
    speed=Column.from_mem_views(speed_data, speed_mask)
    
    return dist,speed

cpdef cpp_traj_sbbox(coor_x,coor_y,len,pos): 
     print("in cpp_coor2traj")
     cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
     cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
     cdef gdf_column* c_len = column_view_from_column(len)
     cdef gdf_column* c_pos = column_view_from_column(pos)
     cdef gdf_column* c_x1 = <gdf_column*>malloc(sizeof(gdf_column))
     cdef gdf_column* c_x2 = <gdf_column*>malloc(sizeof(gdf_column))
     cdef gdf_column* c_y1 = <gdf_column*>malloc(sizeof(gdf_column))
     cdef gdf_column* c_y2 = <gdf_column*>malloc(sizeof(gdf_column))
     
     with nogil:
          traj_sbbox(c_coor_x[0],c_coor_y[0],c_len[0],c_pos[0],c_x1[0],c_y1[0],c_x2[0],c_y2[0])
 
     x1_data, x1_mask = gdf_column_to_column_mem(c_x1)    
     x1=Column.from_mem_views(x1_data,x1_mask)
     x2_data, x2_mask = gdf_column_to_column_mem(c_x2)    
     x2=Column.from_mem_views(x2_data,x2_mask)
     y1_data, y1_mask = gdf_column_to_column_mem(c_y1)    
     y1=Column.from_mem_views(y1_data,y1_mask)
     y2_data, y2_mask = gdf_column_to_column_mem(c_y2)    
     y2=Column.from_mem_views(y2_data,y2_mask)  
    
     return x1,y1,x2,y2