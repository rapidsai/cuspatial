"""
GPU-based coordinate transformation demo: (log/lat)==>(x/y), relative to a camera origin
Note: camera configuration is read from a CSV file using Panda
"""

import numpy as np
from cudf.dataframe import columnops
import cuspatial.bindings.spatial as gis

cam_lon= np.double(-90.66511046)
cam_lat =np.double(42.49197018)

py_lon=[-90.66518941, -90.66540743, -90.66489239]
py_lat=[42.49207437, 42.49202408,42.49266787]
pnt_lon=columnops.as_column(py_lon,dtype=np.float64)

pnt_lat=columnops.as_column(py_lon,dtype=np.float64)
x,y=gis.cpp_ll2coor(cam_lon,cam_lat,pnt_lon,pnt_lat)
x.data.to_array()
y.data.to_array()