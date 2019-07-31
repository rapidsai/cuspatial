"""
GPU-based pip demo using two points from numpy array and 27 polygons read from file
np.binary_repr outputs an integer as a binary string
"""

import numpy as np
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.soa_readers as readers
from cudf.dataframe import columnops

data_dir="/home/jianting/cuspatial/data/"
pnt_x=columnops.as_column(np.array([ -90.666418409895840,-90.665136925928721,-90.671840534675397],dtype=np.float64))
pnt_y=columnops.as_column(np.array([42.492199401857071 ,42.492104092138952,42.490649501411141],dtype=np.float64))
fpos,rpos,plyx,plyy=readers.cpp_read_ply_soa(data_dir+"itsroi.ply")

bm=gis.cpp_pip2_bm(pnt_x,pnt_y,fpos,rpos,plyx,plyy)
bma=bm.data.to_array()

for n in bma:
 np.binary_repr(n, width=fpos.data.size)