"""
GPU-based coordinate transformation demo: (log/lat)==>(x/y), relative to a camera origin
Note: camera configuration is read from a CSV file using Panda
"""

import numpy as np
import pandas as pd
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.traj as traj
import cuspatial.bindings.soa_readers as readers
import cuspatial.utils.traj_utils as tools

data_dir="/home/jianting/cuspatial/data/"
df = pd.read_csv(data_dir+"its_camera_2.csv")
this_cam=df.loc[df["cameraIdString"] == 'HWY_20_AND_LOCUST']
cam_lon= np.double(this_cam.iloc[0]['originLon'])
cam_lat=np.double(this_cam.iloc[0]['originLat'])

pnt_lon,pnt_lat=readers.cpp_read_pnt_lonlat_soa(data_dir+"locust.location");
id=readers.cpp_read_uint_soa(data_dir+"locust.objectid")
ts=readers.cpp_read_ts_soa(data_dir+"locust.time")

#examine binary representatons 
ts_0=ts.data.to_array()[0]
out1=format(ts_0,'016x')
print(out1)
out2=format(ts_0,'064b')
print(out2)

y,m,d,hh,mm,ss,wd,yd,ms,pid=tools.get_ts_struct(ts_0)

pnt_x,pnt_y=gis.cpp_lonlat2coord(cam_lon,cam_lat,pnt_lon,pnt_lat)
num_traj, tid,len,pos=traj.cpp_coord2traj(pnt_x,pnt_y,id,ts)

y,m,d,hh,mm,ss,wd,yd,ms,pid=tools.get_ts_struct(ts_0)

dist,speed=traj.cpp_traj_distspeed(pnt_x,pnt_y,ts,len,pos)
print(dist.data.to_array()[0],speed.data.to_array()[0])

x1,y1,x2,y2=traj.cpp_traj_sbbox(pnt_x,pnt_y,len,pos)
print(x1.data.to_array()[0],x2.data.to_array()[0],y1.data.to_array()[0],y2.data.to_array()[0])
