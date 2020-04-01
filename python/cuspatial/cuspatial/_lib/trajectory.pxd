# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.legacy.cudf cimport *
from libcpp.pair cimport pair

cdef extern from "trajectory.hpp" namespace "cuspatial" nogil:

    cdef size_type derive_trajectories(
        const gdf_column& coor_x,
        const gdf_column& coor_y,
        const gdf_column& pid,
        const gdf_column& ts,
        gdf_column& tid,
        gdf_column& len,
        gdf_column& pos
    ) except +

    cdef pair[gdf_column, gdf_column] trajectory_distance_and_speed(
        const gdf_column& x,
        const gdf_column& y,
        const gdf_column& ts,
        const gdf_column& len,
        const gdf_column& pos
    ) except +

    cdef void trajectory_spatial_bounds(
        const gdf_column& x,
        const gdf_column& y,
        const gdf_column& len,
        const gdf_column& pos,
        gdf_column& bbox_x1,
        gdf_column& bbox_y1,
        gdf_column& bbox_x2,
        gdf_column& bbox_y2
    ) except +

    cdef size_type subset_trajectory_id(
        const gdf_column& id,
        const gdf_column& in_x,
        const gdf_column& in_y,
        const gdf_column& in_id,
        const gdf_column& in_timestamp,
        gdf_column& out_x,
        gdf_column& out_y,
        gdf_column& out_id,
        gdf_column& out_timestamp
    ) except +
