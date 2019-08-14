from cudf.dataframe.column import Column
from cuspatial.bindings.cudf_cpp import * 
from libc.stdlib cimport calloc, malloc, free

cpdef cpp_read_uint_soa(soa_file_name):
	print("in cpp_read_id_soa, reading ",soa_file_name)
	cdef bytes py_bytes = soa_file_name.encode()
	cdef char* c_string = py_bytes
	cdef gdf_column* c_id=<gdf_column*>malloc(sizeof(gdf_column))
	with nogil:
		read_uint_soa(c_string,c_id[0])
	id_data, id_mask = gdf_column_to_column_mem(c_id)
	id=Column.from_mem_views(id_data,id_mask)	
	
	return id

cpdef cpp_read_ts_soa(soa_file_name):
	print("in cpp_read_ts_soa, reading ",soa_file_name)
	cdef bytes py_bytes = soa_file_name.encode()
	cdef char* c_string = py_bytes
	cdef gdf_column* c_ts=<gdf_column*>malloc(sizeof(gdf_column))
	with nogil:
		read_ts_soa(c_string,c_ts[0])
	ts_data, ts_mask = gdf_column_to_column_mem(c_ts)
	ts=Column.from_mem_views(ts_data,ts_mask)	
	
	return ts

cpdef cpp_read_pnt_lonlat_soa(soa_file_name):
	print("in cpp_read_pnt_lonlat_soa, reading ",soa_file_name)
	cdef bytes py_bytes = soa_file_name.encode()
	cdef char* c_string = py_bytes
	cdef gdf_column* c_pnt_lon=<gdf_column*>malloc(sizeof(gdf_column))
	cdef gdf_column* c_pnt_lat=<gdf_column*>malloc(sizeof(gdf_column))
	with nogil:
		read_pnt_lonlat_soa(c_string,c_pnt_lon[0],c_pnt_lat[0])
	lon_data, lon_mask = gdf_column_to_column_mem(c_pnt_lon)
	lon=Column.from_mem_views(lon_data,lon_mask)
	lat_data, lat_mask = gdf_column_to_column_mem(c_pnt_lat)
	lat=Column.from_mem_views(lat_data,lat_mask)
	
	return lon,lat

cpdef cpp_read_pnt_xy_soa(soa_file_name):
	print("in cpp_read_pnt_xy_soa, reading ",soa_file_name)
	cdef bytes py_bytes = soa_file_name.encode()
	cdef char* c_string = py_bytes
	cdef gdf_column* c_pnt_x=<gdf_column*>malloc(sizeof(gdf_column))
	cdef gdf_column* c_pnt_y=<gdf_column*>malloc(sizeof(gdf_column))
	with nogil:
		read_pnt_xy_soa(c_string,c_pnt_x[0],c_pnt_y[0])
	x_data, x_mask = gdf_column_to_column_mem(c_pnt_x)
	x=Column.from_mem_views(x_data,x_mask)
	y_data, y_mask = gdf_column_to_column_mem(c_pnt_y)
	y=Column.from_mem_views(y_data,y_mask)
	
	return x,y

cpdef cpp_read_ply_soa(soa_file_name):
	print("in cpp_read_ply_soa, reading ",soa_file_name)
	cdef bytes py_bytes = soa_file_name.encode()
	cdef char* c_string = py_bytes
	cdef gdf_column* c_ply_fpos=<gdf_column*>malloc(sizeof(gdf_column))
	cdef gdf_column* c_ply_rpos=<gdf_column*>malloc(sizeof(gdf_column))
	cdef gdf_column* c_ply_x=<gdf_column*>malloc(sizeof(gdf_column))
	cdef gdf_column* c_ply_y=<gdf_column*>malloc(sizeof(gdf_column))
	with nogil:
		read_ply_soa(c_string,c_ply_fpos[0],c_ply_rpos[0],c_ply_x[0],c_ply_y[0])
	f_data, f_mask = gdf_column_to_column_mem(c_ply_fpos)
	f_pos=Column.from_mem_views(f_data,f_mask)
	r_data, r_mask = gdf_column_to_column_mem(c_ply_rpos)
	r_pos=Column.from_mem_views(r_data,r_mask)
	x_data, x_mask = gdf_column_to_column_mem(c_ply_x)
	x=Column.from_mem_views(x_data,x_mask)
	y_data, y_mask = gdf_column_to_column_mem(c_ply_y)
	y=Column.from_mem_views(y_data,y_mask)
	
	return f_pos,r_pos,x,y                            
                               
