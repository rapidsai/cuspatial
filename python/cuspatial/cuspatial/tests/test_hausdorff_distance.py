# Copyright (c) 2019, NVIDIA CORPORATION.

"""
A toy example to demonstrate how to convert python arrays into cuSpatial inputs,
invoke the GPU accelerated directed Hausdorff distance computing function in
cuSpatial, convert the results back to python array(s) again to be feed into 
scipy clustering APIs.

For the toy example, by desgin, both AgglomerativeClustering and DBSCAN cluster 
the 2nd and third trajectories into one cluster while leaving the first 
trajectory as the second cluster. 

To run the demo, first install scipy and scikit-learn
by "conda install -c conda-forge scipy scikit-learn"  under cudf_dev environment
"""

import pytest
import numpy as np
import time
import cudf
from cudf.tests.utils import assert_eq
from cudf.core import column
from cuspatial.core import gis

def test_zeros():
    distance = gis.directed_hausdorff_distance(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([1])
    )
    assert cudf.Series(distance)[0] == 0

def test_empty_x():
    with pytest.raises(RuntimeError):
        distance = gis.directed_hausdorff_distance(
            cudf.Series(),
            cudf.Series([0]),
            cudf.Series([0])
        )

def test_empty_y():
    with pytest.raises(RuntimeError):
        distance = gis.directed_hausdorff_distance(
            cudf.Series([0]),
            cudf.Series(),
            cudf.Series([0])
        )

def test_empty_counts():
    with pytest.raises(RuntimeError):
        distance = gis.directed_hausdorff_distance(
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series()
        )

def test_large():
    in_trajs=[]
    in_trajs.append(np.array([[0,0],[1,0]]))
    in_trajs.append(np.array([[0,-1],[1,-1]]))
    out_trajs = np.concatenate([np.asarray(traj) for traj in in_trajs],0)
    py_x=np.array(out_trajs[:,0])
    py_y=np.array(out_trajs[:,1])
    py_cnt = []
    for traj in in_trajs:
     py_cnt.append(len(traj))
    pnt_x = cudf.Series(py_x)
    pnt_y = cudf.Series(py_y)
    cnt = cudf.Series(py_cnt)
    distance=gis.directed_hausdorff_distance(pnt_x,pnt_y,cnt)

    num_set=len(cnt)
    matrix=distance.data.to_array().reshape(num_set,num_set)
    expect = np.array([0, 1, 1, 0])
    assert np.allclose(distance.data.to_array(), expect)

def test_count_one():
    distance = gis.directed_hausdorff_distance(
            cudf.Series([0.0, 0.0]),
            cudf.Series([0.0, 1.0]),
            cudf.Series([1, 1]))
    assert_eq(cudf.Series([0, 1.0, 1, 0]), cudf.Series(distance))

def test_count_two():
    distance = gis.directed_hausdorff_distance(
            cudf.Series([0.0, 0.0, 1.0, 0.0]),
            cudf.Series([0.0, -1.0, 1.0, -1.0]),
            cudf.Series([2, 2]))
    print(cudf.Series(distance))
    assert_eq(cudf.Series([0.0, 1, 1.4142135623730951, 0]),
              cudf.Series(distance))

def test_values():
    in_trajs=[]
    in_trajs.append(np.array([[1,0],[2,1],[3,2],[5,3],[7,1]]))
    in_trajs.append(np.array([[0,3],[2,5],[3,6],[6,5]]))
    in_trajs.append(np.array([[1,4],[3,7],[6,4]]))
    out_trajs = np.concatenate([np.asarray(traj) for traj in in_trajs],0)
    py_x=np.array(out_trajs[:,0])
    py_y=np.array(out_trajs[:,1])
    py_cnt = []
    for traj in in_trajs:
     py_cnt.append(len(traj))
    pnt_x =cudf.Series(py_x)
    pnt_y= cudf.Series(py_y)
    cnt= cudf.Series(py_cnt)
    distance=gis.directed_hausdorff_distance(pnt_x,pnt_y,cnt)

    num_set=len(cnt)
    matrix=distance.data.to_array().reshape(num_set,num_set)
    expect = np.array([
        0, 4.12310563, 4.0,
        3.60555128,  0.0, 1.41421356,
        4.47213595,1.41421356, 0.0])
    assert np.allclose(distance.data.to_array(), expect)

# def test_count_1():
# def test_count_2():
# def test_mismatched_x_y():
# def test_count_greater_than_x():
