# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuspatial.bindings.cudf_cpp cimport *

cdef extern from "traj2.hpp" namespace "cuSpatial" nogil:

   cdef int coor2traj(gdf_column& coor_x,gdf_column& coor_y,gdf_column& pid, gdf_column& ts, gdf_column& tid,gdf_column& len,gdf_column& pos) except + 
   cdef void traj_distspeed(const gdf_column& coor_x,const gdf_column& coor_y,const gdf_column& ts,
 			    const gdf_column& len,const gdf_column& pos,gdf_column& dist,gdf_column& speed) except + 
   cdef void traj_sbbox(const gdf_column& coor_x,const gdf_column& coor_y, const gdf_column& len,const gdf_column& pos,
				gdf_column& bbox_x1,gdf_column& bbox_y1,gdf_column& bbox_x2,gdf_column& bbox_y2) except + 			   
    		
	
    	                                
                                    
                                    
                               