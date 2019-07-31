"""
GPU-based coordinate transformation demo: (log/lat)==>(x/y), relative to a camera origin
Note: camera configuration is read from a CSV file using Panda
"""

import numpy as np
import pandas as pd
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.traj as traj
import cuspatial.bindings.soa_readers as readers
import cuspatial.tests.traj2_test_tools as tools

data_dir="/home/jianting/cuspatial/data/"
df = pd.read_csv(data_dir+"its_camera_2.csv")
this_cam=df.loc[df["cameraIdString"] == 'HWY_20_AND_LOCUST']
cam_x= np.double(this_cam.iloc[0]['originLon'])
cam_y=np.double(this_cam.iloc[0]['originLat'])

pnt_x,pnt_y=readers.cpp_read_pnt_soa(data_dir+"locust.location");
id=readers.cpp_read_id_soa(data_dir+"locust.objectid")
ts=readers.cpp_read_ts_soa(data_dir+"locust.time")
y,m,d,hh,mm,ss,wd,yd,ms,pid=tools.get_ts_struct(ts.data.to_array()[0])

x,y=traj.cpp_ll2coor(cam_x,cam_y,pnt_x,pnt_y)
num_traj, tid,len,pos=traj.cpp_coor2traj(x,y,id,ts)
y,m,d,hh,mm,ss,wd,yd,ms,pid=tools.get_ts_struct(ts.data.to_array()[0])