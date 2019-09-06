from cudf.core.column import Column
from cudf._lib.cudf import *
from libc.stdlib cimport calloc, malloc, free
from libcpp.pair cimport pair

cpdef cpp_read_uint_soa(soa_file_name):
    # print("in cpp_read_id_soa, reading ",soa_file_name)
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column c_id
    with nogil:
        c_id = read_uint32_soa(c_string)
    id_data, id_mask = gdf_column_to_column_mem(&c_id)
    id=Column.from_mem_views(id_data,id_mask)

    return id

cpdef cpp_read_ts_soa(soa_file_name):
    # print("in cpp_read_ts_soa, reading ",soa_file_name)
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column c_ts
    with nogil:
        c_ts = read_timestamp_soa(c_string)
    ts_data, ts_mask = gdf_column_to_column_mem(&c_ts)
    ts=Column.from_mem_views(ts_data,ts_mask)

    return ts

cpdef cpp_read_pnt_lonlat_soa(soa_file_name):
    # print("in cpp_read_pnt_lonlat_soa, reading ",soa_file_name)
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef pair[gdf_column, gdf_column] columns
    with nogil:
        columns = read_lonlat_points_soa(c_string)
    lon_data, lon_mask = gdf_column_to_column_mem(&columns.first)
    lon=Column.from_mem_views(lon_data,lon_mask)
    lat_data, lat_mask = gdf_column_to_column_mem(&columns.second)
    lat=Column.from_mem_views(lat_data,lat_mask)

    return lon,lat

cpdef cpp_read_pnt_xy_soa(soa_file_name):
    # print("in cpp_read_pnt_xy_soa, reading ",soa_file_name)
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef pair[gdf_column, gdf_column] columns
    with nogil:
        columns = read_xy_points_soa(c_string)
    x_data, x_mask = gdf_column_to_column_mem(&columns.first)
    x=Column.from_mem_views(x_data,x_mask)
    y_data, y_mask = gdf_column_to_column_mem(&columns.second)
    y=Column.from_mem_views(y_data,y_mask)

    return x,y

cpdef cpp_read_polygon_soa(soa_file_name):
    # print("in cpp_read_polygon_soa, reading ",soa_file_name)
    cdef bytes py_bytes = soa_file_name.encode()
    cdef char* c_string = py_bytes
    cdef gdf_column* c_ply_fpos=<gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_rpos=<gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_x=<gdf_column*>malloc(sizeof(gdf_column))
    cdef gdf_column* c_ply_y=<gdf_column*>malloc(sizeof(gdf_column))
    with nogil:
        read_polygon_soa(c_string,c_ply_fpos, c_ply_rpos, c_ply_x, c_ply_y)
    f_data, f_mask = gdf_column_to_column_mem(c_ply_fpos)
    f_pos=Column.from_mem_views(f_data,f_mask)
    r_data, r_mask = gdf_column_to_column_mem(c_ply_rpos)
    r_pos=Column.from_mem_views(r_data,r_mask)
    x_data, x_mask = gdf_column_to_column_mem(c_ply_x)
    x=Column.from_mem_views(x_data,x_mask)
    y_data, y_mask = gdf_column_to_column_mem(c_ply_y)
    y=Column.from_mem_views(y_data,y_mask)

    return f_pos,r_pos,x,y

