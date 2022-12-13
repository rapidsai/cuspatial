"""
GPU-based coordinate transformation demo: (log/lat)==>(x/y), relative to a
camera origin

Note: camera configuration is read from a CSV file using Panda
"""

import numpy as np
import pandas as pd

import cuspatial


def get_ts_struct(ts):
    y = ts & 0x3F
    ts = ts >> 6
    m = ts & 0xF
    ts = ts >> 4
    d = ts & 0x1F
    ts = ts >> 5
    hh = ts & 0x1F
    ts = ts >> 5
    mm = ts & 0x3F
    ts = ts >> 6
    ss = ts & 0x3F
    ts = ts >> 6
    wd = ts & 0x8
    ts = ts >> 3
    yd = ts & 0x1FF
    ts = ts >> 9
    ms = ts & 0x3FF
    ts = ts >> 10
    pid = ts & 0x3FF

    return y, m, d, hh, mm, ss, wd, yd, ms, pid


data_dir = "./data/"
df = pd.read_csv(data_dir + "its_camera_2.csv")
this_cam = df.loc[df["cameraIdString"] == "HWY_20_AND_LOCUST"]
cam_lon = np.double(this_cam.iloc[0]["originLon"])
cam_lat = np.double(this_cam.iloc[0]["originLat"])

lonlats = cuspatial.read_points_lonlat(data_dir + "locust.location")
ids = cuspatial.read_uint(data_dir + "locust.objectid")
ts = cuspatial.read_its_timestamps(data_dir + "locust.time")

# examine binary representatons
ts_0 = ts.data.to_array()[0]
out1 = format(ts_0, "016x")
print(out1)
out2 = format(ts_0, "064b")
print(out2)

y, m, d, hh, mm, ss, wd, yd, ms, pid = get_ts_struct(ts_0)

xys = cuspatial.sinusoidal_projection(
    cam_lon, cam_lat, lonlats["lon"], lonlats["lat"]
)
num_traj, trajectories = cuspatial.derive(xys["x"], xys["y"], ids, ts)
#  = num_traj, tid, len, pos =
y, m, d, hh, mm, ss, wd, yd, ms, pid = get_ts_struct(ts_0)
distspeed = cuspatial.distance_and_speed(
    xys["x"], xys["y"], ts, trajectories["length"], trajectories["position"]
)
print(distspeed)

boxes = cuspatial.spatial_bounds(
    xys["x"], xys["y"], trajectories["length"], trajectories["position"]
)
print(boxes.head())
