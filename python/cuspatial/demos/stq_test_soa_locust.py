"""
GPU-based spatial window query demo using 1.3 million points read from file
and (x1,x2,y1,y2)=[-180,180,-90,90] as the query window num should be the same
as x.data.size, both are 1338671
"""

import numpy as np

import cuspatial._lib.soa_readers as readers
import cuspatial._lib.stq as stq

data_dir = "/home/jianting/cuspatial/data/"
pnt_lon, pnt_lat = readers.cpp_read_pnt_lonlat_soa(
    data_dir + "locust.location"
)
num, nlon, nlat = stq.cpp_sw_xy(
    np.double(-180),
    np.double(180),
    np.double(-90),
    np.double(90),
    pnt_lon,
    pnt_lat,
)
print(num)
print(pnt_lon.data.size)
