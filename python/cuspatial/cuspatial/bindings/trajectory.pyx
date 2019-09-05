from cudf import Series, DataFrame
from cudf.core.column import Column
from cudf._lib.cudf import *

from libc.stdlib cimport calloc, malloc, free
from libcpp.pair cimport pair

cpdef cpp_derive_trajectories(x, y, object_id, timestamp):
    x = x.astype('float64')._column
    y = y.astype('float64')._column
    object_id = object_id.astype('int32')._column
    timestamp = timestamp.astype('datetime64[ms]')._column
    cdef gdf_column* c_x = column_view_from_column(x)
    cdef gdf_column* c_y = column_view_from_column(y)
    cdef gdf_column* c_object_id = column_view_from_column(object_id)
    cdef gdf_column* c_timestamp = column_view_from_column(timestamp)
    cdef gdf_column* c_trajectory_id = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_length = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_pos = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
         num_trajectories = derive_trajectories(c_x[0], c_y[0],
                                                c_object_id[0],
                                                c_timestamp[0],
                                                c_trajectory_id[0],
                                                c_length[0], c_pos[0])

    traj_id_data, traj_id_mask = gdf_column_to_column_mem(c_trajectory_id)
    length_data, length_mask = gdf_column_to_column_mem(c_length)
    pos_data, pos_mask = gdf_column_to_column_mem(c_pos)
    trajectory_id = Column.from_mem_views(traj_id_data,
                                          traj_id_mask)
    length = Column.from_mem_views(length_data, length_mask)
    pos = Column.from_mem_views(pos_data, pos_mask)

    return (num_trajectories,
            DataFrame({'trajectory_id': Series(trajectory_id),
                       'length': Series(length),
                       'position': Series(pos)}))


cpdef cpp_trajectory_distance_and_speed(x, y, timestamp, length, pos):
    x = x.astype('float64')._column
    y = y.astype('float64')._column
    timestamp = timestamp.astype('datetime64[ms]')._column
    length = length.astype('int32')._column
    pos = pos.astype('int32')._column
    cdef gdf_column* c_x = column_view_from_column(x)
    cdef gdf_column* c_y = column_view_from_column(y)
    cdef gdf_column* c_timestamp = column_view_from_column(timestamp)
    cdef gdf_column* c_length = column_view_from_column(length)
    cdef gdf_column* c_pos = column_view_from_column(pos)
    cdef pair[gdf_column, gdf_column] c_distance_speed

    with nogil:
        c_distance_speed = trajectory_distance_and_speed(c_x[0], c_y[0],
                                                         c_timestamp[0],
                                                         c_length[0],c_pos[0])

    dist_data, dist_mask = gdf_column_to_column_mem(&c_distance_speed.first)
    speed_data, speed_mask = gdf_column_to_column_mem(&c_distance_speed.second)
    dist=Column.from_mem_views(dist_data, dist_mask)
    speed=Column.from_mem_views(speed_data, speed_mask)

    return Series(dist), Series(speed)

cpdef cpp_trajectory_spatial_bounds(coor_x,coor_y,length,pos):
    coor_x = coor_x.astype('float64')._column
    coor_y = coor_y.astype('float64')._column
    length = length.astype('int32')._column
    pos = pos.astype('int32')._column
    cdef gdf_column* c_coor_x = column_view_from_column(coor_x)
    cdef gdf_column* c_coor_y = column_view_from_column(coor_y)
    cdef gdf_column* c_length = column_view_from_column(length)
    cdef gdf_column* c_pos = column_view_from_column(pos)
    cdef gdf_column* c_x1 = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_x2 = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_y1 = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_y2 = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        trajectory_spatial_bounds(c_coor_x[0], c_coor_y[0],
                                c_length[0], c_pos[0],
                                c_x1[0], c_y1[0], c_x2[0], c_y2[0])

    x1_data, x1_mask = gdf_column_to_column_mem(c_x1)
    x1 = Column.from_mem_views(x1_data,x1_mask)
    x2_data, x2_mask = gdf_column_to_column_mem(c_x2)
    x2 = Column.from_mem_views(x2_data,x2_mask)
    y1_data, y1_mask = gdf_column_to_column_mem(c_y1)
    y1 = Column.from_mem_views(y1_data,y1_mask)
    y2_data, y2_mask = gdf_column_to_column_mem(c_y2)
    y2 = Column.from_mem_views(y2_data,y2_mask)

    return DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

cpdef cpp_subset_trajectory_id(id, in_x, in_y, in_id, in_timestamp):
    cdef gdf_column* c_id = column_view_from_column(id)
    cdef gdf_column* c_in_x = column_view_from_column(in_x)
    cdef gdf_column* c_in_y = column_view_from_column(in_y)
    cdef gdf_column* c_in_id = column_view_from_column(in_id)
    cdef gdf_column* c_in_timestamp = column_view_from_column(in_timestamp)

    cdef gdf_column* c_out_x = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_out_y = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_out_id = <gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_out_timestamp = <gdf_column*>malloc(sizeof(gdf_column))

    with nogil:
        count = subset_trajectory_id(c_id[0], c_in_x[0], c_in_y[0], c_in_id[0], 
                                     c_in_timestamp[0], c_out_x[0], c_out_y[0],
                                     c_out_id[0], c_out_timestamp[0])
    
    x_data, x_mask = gdf_column_to_column_mem(c_out_x)
    x = Column.from_mem_views(x_data,x_mask)
    y_data, y_mask = gdf_column_to_column_mem(c_out_y)
    y = Column.from_mem_views(y_data,y_mask)
    id_data, id_mask = gdf_column_to_column_mem(c_out_id)
    id = Column.from_mem_views(id_data,id_mask)
    timestamp_data, timestamp_mask = gdf_column_to_column_mem(c_out_timestamp)
    timestamp = Column.from_mem_views(timestamp_data, timestamp_mask)

    return x, y, id, timestamp
