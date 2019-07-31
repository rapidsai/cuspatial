# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuspatial.bindings.cudf_cpp cimport *

cdef extern from "stq2.hpp" namespace "cuSpatial" nogil:

   cdef int sw_xy(const gdf_scalar x1,const gdf_scalar x2,const gdf_scalar y1,const gdf_scalar y2,
   	const gdf_column  & in_x,const gdf_column  & in_y,gdf_column & out_x,gdf_column & out_y) except +	                               
  		   
   	
    	                                
                                    
                                    
                               