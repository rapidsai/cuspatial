# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from libcpp.pair cimport pair

cdef extern from "query.hpp" namespace "cuspatial" nogil:

    cdef pair[gdf_column, gdf_column] spatial_window_points(const gdf_scalar* x1,
                                                            const gdf_scalar* x2,
                                                            const gdf_scalar* y1,
                                                            const gdf_scalar* y2,
                                                            const gdf_column* in_x,
                                                            const gdf_column* in_y) except +
