"""
GPU-based demos for computing trajectories from x/y coordinates (after transformation),
and then computing distance(length)/speed and spaital bounding boxes of the trajectories
Note: camera configuration is read from a CSV file using Panda
"""

import numpy as np
import pandas as pd
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.traj as traj
import cuspatial.bindings.soa_readers as readers
import cuspatial.tests.traj2_test_tools as tools
from cudf.dataframe import columnops
import cudf

data_dir="/home/jianting/cuspatial/data/"
df = pd.read_csv(data_dir+"its_camera_2.csv")
this_cam=df.loc[df["cameraIdString"] == 'HWY_20_AND_LOCUST']
cam_x= np.double(this_cam.iloc[0]['originLon'])
cam_y=np.double(this_cam.iloc[0]['originLat'])

pnt_lon,pnt_lat=readers.cpp_read_pnt_lonlat_soa(data_dir+"locust.location");
id=readers.cpp_read_uint_soa(data_dir+"locust.objectid")
ts=readers.cpp_read_ts_soa(data_dir+"locust.time")
x,y=gis.cpp_ll2coor(cam_x,cam_y,pnt_lon,pnt_lat)

num_traj, tid,len,pos=traj.cpp_coor2traj(x,y,id,ts)
dist,speed=traj.cpp_traj_distspeed(x,y,ts,len,pos)
print(dist.data.to_array()[0],speed.data.to_array()[0])

x1,y1,x2,y2=traj.cpp_traj_sbbox(x,y,len,pos)
print(x1.data.to_array()[0],x2.data.to_array()[0],y1.data.to_array()[0],y2.data.to_array()[0])
