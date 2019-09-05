# Copyright (c) 2019, NVIDIA CORPORATION.

"""
GPU-based coordinate transformation demo: (log/lat)==>(x/y), relative to a camera origin
Note: camera configuration is read from a CSV file using Panda
"""

import pytest
import cudf
from cudf.tests.utils import assert_eq
import numpy as np
import pandas as pd
import cuspatial.bindings.spatial as gis
import cuspatial.bindings.trajectory as traj
import cuspatial.bindings.soa_readers as readers
import cuspatial.utils.traj_utils as tools

def test_derive_trajectories_zeros():
    num_trajectories = traj.cpp_derive_trajectories(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
    )
    assert num_trajectories[0] == 1
    assert_eq(num_trajectories[1], cudf.DataFrame({
        'trajectory_id': cudf.Series([0]).astype('int32'),
        'length': cudf.Series([1]).astype('int32'),
        'position': cudf.Series([1]).astype('int32'),
    }))

def test_derive_trajectories_ones():
    num_trajectories = traj.cpp_derive_trajectories(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
    )
    assert num_trajectories[0] == 1
    assert_eq(num_trajectories[1], cudf.DataFrame({
        'trajectory_id': cudf.Series([1]).astype('int32'),
        'length': cudf.Series([1]).astype('int32'),
        'position': cudf.Series([1]).astype('int32'),
    }))

def test_derive_trajectories_two():
    num_trajectories = traj.cpp_derive_trajectories(
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
    )
    assert num_trajectories[0] == 2
    assert_eq(num_trajectories[1], cudf.DataFrame({
        'trajectory_id': cudf.Series([0, 1]).astype('int32'),
        'length': cudf.Series([1, 1]).astype('int32'),
        'position': cudf.Series([1, 2]).astype('int32')
    }))

def test_derive_trajectories_many():
    np.random.seed(0)
    num_trajectories = traj.cpp_derive_trajectories(
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
        cudf.Series(np.random.randint(0, 10, 10)),
    )
    assert num_trajectories[0] == 6
    assert_eq(num_trajectories[1], cudf.DataFrame({
        'trajectory_id': cudf.Series([0, 3, 4, 5, 8, 9]).astype('int32'),
        'length': cudf.Series([2, 2, 1, 2, 1, 2]).astype('int32'),
        'position': cudf.Series([2, 4, 5, 7, 8, 10]).astype('int32'),
    }))

def test_trajectory_distance_and_speed_zeros():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
    )
    assert_eq(dist, cudf.Series([-1.0]))
    assert_eq(speed, cudf.Series([-1.0]))

def test_trajectory_distance_and_speed_ones():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
    )
    assert_eq(dist, cudf.Series([-1.0]))
    assert_eq(speed, cudf.Series([-1.0]))

def test_one_trajectory_one_meter_one_second():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([0.0, 1.0]),
        cudf.Series([0.0, 0.0]),
        cudf.Series([0, 1]),
        cudf.Series([2]),
        cudf.Series([2]),
    )
    assert_eq(dist, cudf.Series([1000.0]))
    assert_eq(speed, cudf.Series([1.0]))

def test_trajectory_distance_and_speed_single_trajectory():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([
            1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0]),
        cudf.Series([
            0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0]),
        cudf.Series([
            1000, 2000, 3000, 4000, 5000, 6000, 7000,
            8000, 9000, 10000, 11000, 12000]),
        cudf.Series([5, 4, 3]),
        cudf.Series([5, 9, 12]),
    )
    assert_eq(dist, cudf.Series([7892.922363, -1.0, 5064.495117]))
    assert_eq(speed, cudf.Series([1.973231, -1.0, 0.633062]))

def test_trajectory_distance_and_speed_full():
    data_dir = "/home/jianting/cuspatial/data/"
    df = pd.read_csv(data_dir + "its_camera_2.csv")
    this_cam = df.loc[df["cameraIdString"] == "HWY_20_AND_LOCUST"]
    cam_lon = np.double(this_cam.iloc[0]["originLon"])
    cam_lat = np.double(this_cam.iloc[0]["originLat"])

    pnt_lon, pnt_lat = readers.cpp_read_pnt_lonlat_soa(data_dir + "locust.location")
    id = readers.cpp_read_uint_soa(data_dir + "locust.objectid")
    ts = readers.cpp_read_ts_soa(data_dir + "locust.time")

# examine binary representatons
    ts_0 = ts.data.to_array()[0]
    out1 = format(ts_0, "016x")
    print(out1)
    out2 = format(ts_0, "064b")
    print(out2)

    y, m, d, hh, mm, ss, wd, yd, ms, pid = tools.get_ts_struct(ts_0)

    pnt_x, pnt_y = gis.cpp_lonlat2coord(cam_lon, cam_lat, pnt_lon, pnt_lat)
    num_traj, tid, len, pos = traj.cpp_coord2traj(pnt_x, pnt_y, id, ts)

    y, m, d, hh, mm, ss, wd, yd, ms, pid = tools.get_ts_struct(ts_0)

    dist, speed = traj.cpp_traj_distspeed(pnt_x, pnt_y, ts, len, pos)
    print(dist.data.to_array()[0], speed.data.to_array()[0])

    x1, y1, x2, y2 = traj.cpp_traj_sbbox(pnt_x, pnt_y, len, pos)
    print(
        x1.data.to_array()[0],
        x2.data.to_array()[0],
        y1.data.to_array()[0],
        y2.data.to_array()[0],
    )
