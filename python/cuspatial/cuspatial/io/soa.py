# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import Series, DataFrame
from cuspatial._lib.soa_readers import (
    cpp_read_uint_soa,
    cpp_read_ts_soa,
    cpp_read_pnt_lonlat_soa,
    cpp_read_pnt_xy_soa,
    cpp_read_polygon_soa
)

def read_uint(filename):
    return Series(cpp_read_uint_soa(filename))

def read_ts(filename):
    return Series(cpp_read_ts_soa(filename))

def read_points_lonlat(filename):
    result = cpp_read_pnt_lonlat_soa(filename)
    return DataFrame({'lon': result[0],
                      'lat': result[1]
    })

def read_points_xy_km(filename):
    result = cpp_read_pnt_xy_soa(filename)
    return DataFrame({'x': result[0],
                      'y': result[1]
    })

def read_polygon(filename):
    result = cpp_read_polygon_soa(filename)
    return DataFrame({'f_pos': result[0],
                      'r_pos': result[1],
                      'x': result[2],
                      'y': result[3]
    })
