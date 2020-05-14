# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.legacy.cudf cimport *
from libcpp.pair cimport pair


cdef extern from "legacy/query.hpp" namespace "cuspatial" nogil:
    cdef pair[gdf_column, gdf_column] spatial_window_points(
        const gdf_scalar& left,
        const gdf_scalar& bottom,
        const gdf_scalar& right,
        const gdf_scalar& top,
        const gdf_column& x,
        const gdf_column& y
    ) except +
