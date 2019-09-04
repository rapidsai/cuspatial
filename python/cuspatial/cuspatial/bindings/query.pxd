# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from libcpp.pair cimport pair

cdef extern from "query.hpp" namespace "cuspatial" nogil:

    cdef pair[gdf_column, gdf_column] spatial_window_points(
        const gdf_scalar& left,
        const gdf_scalar& bottom,
        const gdf_scalar& right,
        const gdf_scalar& top,
        const gdf_column& x,
        const gdf_column& y) except +
