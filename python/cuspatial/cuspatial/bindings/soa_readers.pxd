# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
# distutils: include_dirs = cuspatial/bindings/

from cuspatial.bindings.cudf_cpp cimport *

cdef extern from "soa_readers.hpp" namespace "cuSpatial" nogil:
	cdef void read_id_soa(const char *pnt_fn,gdf_column& id) except +           
	cdef void read_ts_soa(const char *ts_fn, gdf_column& ts) except +   
	cdef void read_pnt_soa(const char *pnt_fn, gdf_column& pnt_x,gdf_column& pnt_y) except +   
	cdef void read_ply_soa(const char *ply_fn,gdf_column& ply_fpos, gdf_column& ply_rpos,
                                   gdf_column& ply_x,gdf_column& ply_y) except +   
                       
                                    
                                    
                                    
                               
