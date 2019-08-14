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

cdef extern from "coor_trans.hpp" namespace "cuSpatial" nogil: 
 cdef void ll2coor(const gdf_scalar cam_x,const gdf_scalar cam_y,const gdf_column  & in_x,const gdf_column  & in_y,
   	gdf_column & out_x,gdf_column & out_y) except +	                               

cdef extern from "haversine.hpp" namespace "cuSpatial" nogil:  
 gdf_column haversine_distance(const gdf_column& x1,const gdf_column& y1,
 	const gdf_column& x2,const gdf_column& y2)except +

cdef extern from "hausdorff.hpp" namespace "cuSpatial" nogil:  
 gdf_column& hausdorff_distance(const gdf_column& coor_x,const gdf_column& coor_y,
	const gdf_column& cnt)except + 

                                    
                                    
                                    
                               
