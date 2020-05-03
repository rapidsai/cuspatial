# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.legacy.cudf cimport *
from libcpp.pair cimport pair

cdef extern from "legacy/point_in_polygon.hpp" namespace "cuspatial" nogil:
    cdef gdf_column point_in_polygon_bitmap(
        const gdf_column& pnt_x,
        const gdf_column& pnt_y,
        const gdf_column& ply_fpos,
        const gdf_column& ply_rpos,
        const gdf_column& ply_x,
        const gdf_column& ply_y
    ) except +

cdef extern from "legacy/coordinate_transform.hpp" namespace "cuspatial" nogil:
    cdef pair[gdf_column, gdf_column] lonlat_to_coord(
        const gdf_scalar& cam_x,
        const gdf_scalar& cam_y,
        const gdf_column& in_x,
        const gdf_column& in_y
    ) except +

cdef extern from "haversine.hpp" namespace "cuspatial" nogil:
    gdf_column haversine_distance(
        const gdf_column& x1,
        const gdf_column& y1,
        const gdf_column& x2,
        const gdf_column& y2
    ) except +

cdef extern from "legacy/hausdorff.hpp" namespace "cuspatial" nogil:
    gdf_column& directed_hausdorff_distance(
        const gdf_column& coor_x,
        const gdf_column& coor_y,
        const gdf_column& cnt
    ) except +

cdef extern from "query.hpp" namespace "cuspatial" nogil:
    cdef pair[gdf_column, gdf_column] spatial_window_points(
        const gdf_scalar& left,
        const gdf_scalar& bottom,
        const gdf_scalar& right,
        const gdf_scalar& top,
        const gdf_column& x,
        const gdf_column& y
    ) except +
