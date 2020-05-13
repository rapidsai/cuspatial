# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.legacy.cudf cimport *
from libcpp.pair cimport pair

cdef extern from "legacy/haversine.hpp" namespace "cuspatial" nogil:
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

cdef extern from "legacy/query.hpp" namespace "cuspatial" nogil:
    cdef pair[gdf_column, gdf_column] spatial_window_points(
        const gdf_scalar& left,
        const gdf_scalar& bottom,
        const gdf_scalar& right,
        const gdf_scalar& top,
        const gdf_column& x,
        const gdf_column& y
    ) except +
