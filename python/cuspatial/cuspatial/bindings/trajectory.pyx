from cudf.core.column import Column
from cudf._lib.cudf import *

from libc.stdlib cimport calloc, malloc, free
from libcpp.pair cimport pair
                         
cpdef cpp_derive_trajectories(x, y, object_id, timestamp):
    cdef gdf_column* c_x = column_view_from_column(x)
    cdef gdf_column* c_y = column_view_from_column(y)
    cdef gdf_column* c_object_id = column_view_from_column(object_id)
    cdef gdf_column* c_timestamp = column_view_from_column(timestamp)
    cdef gdf_column* c_trajectory_id = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_len = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_pos = <gdf_column*>malloc(sizeof(gdf_column))
    
    with nogil:
         num_trajectories = derive_trajectories(c_x[0], c_y[0],
                                                c_object_id[0],
                                                c_timestamp[0],
                                                c_trajectory_id[0],
                                                c_len[0], c_pos[0])

    traj_id_data, traj_id_mask = gdf_column_to_column_mem(c_trajectory_id)
    len_data, len_mask = gdf_column_to_column_mem(c_len)    
    pos_data, pos_mask = gdf_column_to_column_mem(c_pos)
    trajectory_id = Column.from_mem_views(traj_id_data,
                                          traj_id_mask)
    len = Column.from_mem_views(len_data, len_mask)
    pos = Column.from_mem_views(pos_data, pos_mask)
    
    return num_trajectories, trajectory_id, len, pos

cpdef cpp_trajectory_distance_and_speed(x, y, timestamp, len, pos):
    cdef gdf_column* c_x = column_view_from_column(x)
    cdef gdf_column* c_y = column_view_from_column(y)
    cdef gdf_column* c_timestamp = column_view_from_column(timestamp)
    cdef gdf_column* c_len = column_view_from_column(len)
    cdef gdf_column* c_pos = column_view_from_column(pos)
    cdef pair[gdf_column, gdf_column] c_distance_speed

    with nogil:
        c_distance_speed = trajectory_distance_and_speed(c_x[0], c_y[0],
                                                         c_timestamp[0],
                                                         c_len[0],c_pos[0])

    dist_data, dist_mask = gdf_column_to_column_mem(&c_distance_speed.first)
    speed_data, speed_mask = gdf_column_to_column_mem(&c_distance_speed.second)
    dist=Column.from_mem_views(dist_data, dist_mask)
    speed=Column.from_mem_views(speed_data, speed_mask)
    
    return dist,speed

cpdef cpp_trajectory_spatial_bounds(coor_x,coor_y,len,pos):
     cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
     cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
     cdef gdf_column* c_len = column_view_from_column(len)
     cdef gdf_column* c_pos = column_view_from_column(pos)
     cdef gdf_column* c_x1 = <gdf_column*>malloc(sizeof(gdf_column))
     cdef gdf_column* c_x2 = <gdf_column*>malloc(sizeof(gdf_column))
     cdef gdf_column* c_y1 = <gdf_column*>malloc(sizeof(gdf_column))
     cdef gdf_column* c_y2 = <gdf_column*>malloc(sizeof(gdf_column))
     
     with nogil:
          trajectory_spatial_bounds(c_coor_x[0], c_coor_y[0],
                                    c_len[0], c_pos[0],
                                    c_x1[0], c_y1[0], c_x2[0], c_y2[0])
 
     x1_data, x1_mask = gdf_column_to_column_mem(c_x1)    
     x1=Column.from_mem_views(x1_data,x1_mask)
     x2_data, x2_mask = gdf_column_to_column_mem(c_x2)    
     x2=Column.from_mem_views(x2_data,x2_mask)
     y1_data, y1_mask = gdf_column_to_column_mem(c_y1)    
     y1=Column.from_mem_views(y1_data,y1_mask)
     y2_data, y2_mask = gdf_column_to_column_mem(c_y2)    
     y2=Column.from_mem_views(y2_data,y2_mask)  
    
     return x1,y1,x2,y2
