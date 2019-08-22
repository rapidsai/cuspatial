"""
GPU-accelerated Haversine distance computation among three cities: New York, Paris and Sydney
Results match https://www.vcalc.com/wiki/vCalc/Haversine+-+Distance
Note: make sure cudf_dev conda environment is activated
"""

import numpy as np
import time
from cudf.dataframe import columnops
import cuspatial.bindings.spatial as gis

cities=[]
cities.append(np.array(['New York', -74.0060,40.7128],dtype=object))
cities.append(np.array(['Paris', 2.3522,48.8566],dtype=object))
cities.append(np.array(['Sydney', 151.2093,-33.8688],dtype=object))
pnt_x1=[]
pnt_y1=[]
pnt_x2=[]
pnt_y2=[]
for i in range(len(cities)):
 for j in range(len(cities)):
  pnt_x1.append(cities[i][1])
  pnt_y1.append(cities[i][2])
  pnt_x2.append(cities[j][1])
  pnt_y2.append(cities[j][2])
x1=columnops.as_column(pnt_x1,dtype=np.float64)
y1=columnops.as_column(pnt_y1,dtype=np.float64)
x2=columnops.as_column(pnt_x2,dtype=np.float64)
y2=columnops.as_column(pnt_y2,dtype=np.float64)
dis=gis.cpp_haversine_distance(x1,y1,x2,y2)
dis.data.to_array().reshape(3,3)

