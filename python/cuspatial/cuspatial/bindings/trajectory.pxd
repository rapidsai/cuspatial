# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from libcpp.pair cimport pair

cdef extern from "trajectory.hpp" namespace "cuspatial" nogil:

   cdef int derive_trajectories(gdf_column& coor_x,
                                gdf_column& coor_y,
                                gdf_column& pid,
                                gdf_column& ts,
                                gdf_column& tid,
                                gdf_column& len,
                                gdf_column& pos) except +

   cdef pair[gdf_column, gdf_column] trajectory_distance_and_speed(
      const gdf_column& x,
      const gdf_column& y,
      const gdf_column& ts,
      const gdf_column& len,
      const gdf_column& pos
  ) except +

   cdef void trajectory_spatial_bounds(const gdf_column& x,
                                       const gdf_column& y,
                                       const gdf_column& len,
                                       const gdf_column& pos,
                                       gdf_column& bbox_x1,
                                       gdf_column& bbox_y1,
                                       gdf_column& bbox_x2,
                                       gdf_column& bbox_y2) except +
