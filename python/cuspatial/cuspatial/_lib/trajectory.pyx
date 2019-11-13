# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf import Series, DataFrame
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
        num_trajectories = derive_trajectories(
            c_x[0], c_y[0],
            c_object_id[0],
            c_timestamp[0],
            c_trajectory_id[0],
            c_length[0], c_pos[0]
        )

    trajectory_id = gdf_column_to_column(c_trajectory_id)
    length = gdf_column_to_column(c_length)
    pos = gdf_column_to_column(c_pos)

    return DataFrame(
        {
            'trajectory_id': Series(trajectory_id),
            'length': Series(length),
            'position': Series(pos)
        }
    )


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
                                                         c_length[0], c_pos[0])

    dist = gdf_column_to_column(&c_distance_speed.first)
    speed = gdf_column_to_column(&c_distance_speed.second)

    return Series(dist), Series(speed)

cpdef cpp_trajectory_spatial_bounds(coor_x, coor_y, length, pos):
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
        trajectory_spatial_bounds(
            c_coor_x[0],
            c_coor_y[0],
            c_length[0],
            c_pos[0],
            c_x1[0],
            c_y1[0],
            c_x2[0],
            c_y2[0]
        )

    x1 = gdf_column_to_column(c_x1)
    x2 = gdf_column_to_column(c_x2)
    y1 = gdf_column_to_column(c_y1)
    y2 = gdf_column_to_column(c_y2)

    return DataFrame({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

cpdef cpp_subset_trajectory_id(ids, in_x, in_y, in_id, in_timestamp):
    ids = ids.astype('int32')._column
    in_x = in_x.astype('float64')._column
    in_y = in_y.astype('float64')._column
    in_id = in_id.astype('int32')._column
    in_timestamp = in_timestamp.astype('datetime64[ms]')._column
    cdef gdf_column* c_id = column_view_from_column(ids)
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

    x = gdf_column_to_column(c_out_x)
    y = gdf_column_to_column(c_out_y)
    ids = gdf_column_to_column(c_out_id)
    timestamp = gdf_column_to_column(c_out_timestamp)

    return DataFrame({'x': Series(x),
                      'y': Series(y),
                      'ids': Series(ids),
                      'timestamp': Series(timestamp)})
