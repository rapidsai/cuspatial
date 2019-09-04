"""
GPU accelerated coordinate transformation test: (log/lat)==>(x/y), relative to a camera origin

Note:  make sure cudf_dev conda environment is activated
"""

import cudf
import numpy as np
import cudf.core.column as column
import cuspatial.bindings.spatial as gis

cam_lon= np.double(-90.66511046)
cam_lat =np.double(42.49197018)

py_lon=[-90.66518941, -90.66540743, -90.66489239]
py_lat=[42.49207437, 42.49202408,42.49266787]
pnt_lon=column.as_column(py_lon,dtype=np.float64)
pnt_lat=column.as_column(py_lat,dtype=np.float64)

#note: x/y coordinates in killometers -km 
x,y=gis.cpp_lonlat2coord(cam_lon,cam_lat,pnt_lon,pnt_lat)
x.data.to_array()
y.data.to_array()
print(cudf.Series(x))
print(cudf.Series(y))
