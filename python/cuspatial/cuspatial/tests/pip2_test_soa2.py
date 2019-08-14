"""
GPU-based pip demo using 1.3 million points and 27 polygons, both are read from file
np.binary_repr outputs an integer as a binary string
"""

import numpy as np
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.soa_readers as readers
from cudf.dataframe import columnops

data_dir="/home/jianting/cuspatial/data/"
pnt_lon,pnt_lat=readers.cpp_read_pnt_lonlat_soa(data_dir+"locust.location");
fpos,rpos,plyx,plyy=readers.cpp_read_ply_soa(data_dir+"itsroi.ply")

bm=gis.cpp_pip2_bm(pnt_lon,pnt_lat,fpos,rpos,plyx,plyy)
bma=bm.data.to_array()
for n in bma:
 np.binary_repr(n, width=fpos.data.size)