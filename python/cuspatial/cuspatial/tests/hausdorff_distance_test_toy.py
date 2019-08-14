"""
Note: make sure cudf_dev conda environment is activated
"""

import numpy as np
import time
from cudf.dataframe import columnops
import cuspatial.bindings.spatial as gis

pnt_x=columnops.as_column(np.array([0,-8,6],dtype=np.float64))
pnt_y=columnops.as_column(np.array([0,-8,6],dtype=np.float64))
cnt=columnops.as_column(np.array([1,2],dtype=np.int32))

matrix=gis.cpp_directed_hausdorff(pnt_x,pnt_y,cnt)
