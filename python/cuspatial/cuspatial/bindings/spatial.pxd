# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuspatial.bindings.cudf_cpp cimport *

cdef extern from "pip2.hpp" namespace "cuSpatial" nogil:

   cdef gdf_column pip2_bm(const gdf_column& pnt_x,const gdf_column& pnt_y,
                                   const gdf_column& ply_fpos, const gdf_column& ply_rpos,
                                   const gdf_column& ply_x,const gdf_column& ply_y) except +
                                    
                                    
                                    
                                    
                               
