"""
demo of reading camera configuration into a Panda dataframe, 
reading identify (id) and timestamp (ts) columns from respective files,
and formatting timestamp as a hexadecimal and a binary string, respectively. 
This is useful to verify the bit order and the correctness of timestamp as a userdefined struct,
which is physically represented as a int64_t
fileds defined in the Tim struct can be accessed through a simple tool (cuspatial.tests.traj2_test_tools as tools)
Note: do not forget to add {CUSPATIAL_HOME}/pyton/cuspatial to your PYTHONPATH if it is not already there
"""

import numpy as np
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.traj as traj
import cuspatial.bindings.soa_readers as readers
import cuspatial.tests.traj2_test_tools as tools
import pandas as pd

data_dir="/home/jianting/cuspatial/data/"
#df = cudf.read_csv(data_dir+"its_camera_2.csv")
df = pd.read_csv(data_dir+"its_camera_2.csv")
this_cam=df.loc[df["cameraIdString"] == 'HWY_20_AND_LOCUST']
cam_x= np.double(this_cam.iloc[0]['originLon'])
cam_y=np.double(this_cam.iloc[0]['originLat'])

pnt_x,pnt_y=readers.cpp_read_pnt_soa(data_dir+"locust.location");
id=readers.cpp_read_id_soa(data_dir+"locust.objectid")
ts=readers.cpp_read_ts_soa(data_dir+"locust.time")

out1=format(ts.data.to_array()[0],'016x')
print(out1)
out2=format(ts.data.to_array()[0],'064b')
print(out2)

y,m,d,hh,mm,ss,wd,yd,ms,pid=tools.get_ts_struct(ts.data.to_array()[0])
print(y,m,d,hh,mm,ss,wd,yd,ms,pid)