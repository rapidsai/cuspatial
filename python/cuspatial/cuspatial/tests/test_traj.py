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
    assert_eq(dist, cudf.Series([-2.0]))
    assert_eq(speed, cudf.Series([-2.0]))

def test_trajectory_distance_and_speed_ones():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
    )
    assert_eq(dist, cudf.Series([-2.0]))
    assert_eq(speed, cudf.Series([-2.0]))

def test_one_trajectory_one_meter_one_second():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([0.0, 0.001]),
        cudf.Series([0.0, 0.0]),
        cudf.Series([0, 1000]),
        cudf.Series([2]),
        cudf.Series([2]),
    )
    assert_eq(dist, cudf.Series([1.0]))
    assert_eq(speed, cudf.Series([1.0]))

def test_two_trajectories_one_meter_one_second():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([0.0, 0.001, 0.0, 0.0]),
        cudf.Series([0.0, 0.0, 0.0, 0.001]),
        cudf.Series([0, 1000, 0, 1000]),
        cudf.Series([2, 2]),
        cudf.Series([2, 4]),
    )
    assert_eq(dist, cudf.Series([1.0, 1.0]))
    assert_eq(speed, cudf.Series([1.0, 1.0]))

def test_trajectory_distance_and_speed_single_trajectory():
    dist, speed = traj.cpp_trajectory_distance_and_speed(
        cudf.Series([
            1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0]),
        cudf.Series([
            0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0]),
        cudf.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        cudf.Series([5, 4, 3]),
        cudf.Series([5, 9, 12]),
    )
    assert_eq(dist, cudf.Series([7892.922363, 6812.55908203125, 8485.28125]))
    assert_eq(speed, cudf.Series([1973230.625, 2270853., 4242640.5])) # fast!

