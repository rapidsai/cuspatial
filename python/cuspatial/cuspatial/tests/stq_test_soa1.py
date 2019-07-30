"""
GPU-based spatial window query demo using 1.3 million points read from file
and (x1,x2,y1,y2)=[-180,180,-90,90] as the query window 
num should be the same as x.data.size, both are 1338671
"""

import numpy as np
import pandas as pd
import cuspatial.bindings.traj as traj
import cuspatial.bindings.stq as stq
import cuspatial.bindings.soa_readers as readers
from cudf.dataframe import columnops
import cudf

data_dir="/home/jianting/cuspatial/data/"
x,y=readers.cpp_read_pnt_soa(data_dir+"locust.location");
num,nx,ny=stq.cpp_sw_xy(np.double(-180),np.double(180),np.double(-90),np.double(90),x,y)
print(num)
print(x.data.size)

	
