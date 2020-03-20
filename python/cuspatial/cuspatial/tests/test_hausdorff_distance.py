# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial

""" These tests and possibly others are creating a segfault
def test_zeros():
    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series([0.0]), cudf.Series([0.0]), cudf.Series([1])
    )
    assert_eq(distance, cudf.DataFrame([0.0]))


def test_empty_x():
    with pytest.raises(RuntimeError):
        distance = cuspatial.directed_hausdorff_distance(  # noqa: F841
            cudf.Series(), cudf.Series([0]), cudf.Series([0])
        )


def test_empty_y():
    with pytest.raises(RuntimeError):
        distance = cuspatial.directed_hausdorff_distance(  # noqa: F841
            cudf.Series([0]), cudf.Series(), cudf.Series([0])
        )


def test_empty_counts():
    with pytest.raises(RuntimeError):
        distance = cuspatial.directed_hausdorff_distance(  # noqa: F841
            cudf.Series([0]), cudf.Series([0]), cudf.Series()
        )
"""

def test_large():
    in_trajs = []
    in_trajs.append(np.array([[0, 0], [1, 0]]))
    in_trajs.append(np.array([[0, -1], [1, -1]]))
    out_trajs = np.concatenate([np.asarray(traj) for traj in in_trajs], 0)
    py_x = np.array(out_trajs[:, 0])
    py_y = np.array(out_trajs[:, 1])
    py_cnt = []
    for traj in in_trajs:
        py_cnt.append(len(traj))
    pnt_x = cudf.Series(py_x)
    pnt_y = cudf.Series(py_y)
    cnt = cudf.Series(py_cnt)
    distance = cuspatial.directed_hausdorff_distance(pnt_x, pnt_y, cnt)

    assert_eq(distance, cudf.DataFrame({0: [0, 1.0], 1: [1.0, 0]}))


def test_count_one():
    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series([0.0, 0.0]), cudf.Series([0.0, 1.0]), cudf.Series([1, 1])
    )
    assert_eq(distance, cudf.DataFrame({0: [0, 1.0], 1: [1.0, 0]}))


def test_count_two():
    distance = cuspatial.directed_hausdorff_distance(
        cudf.Series([0.0, 0.0, 1.0, 0.0]),
        cudf.Series([0.0, -1.0, 1.0, -1.0]),
        cudf.Series([2, 2]),
    )
    assert_eq(
        distance, cudf.DataFrame({0: [0, 1.4142135623730951], 1: [1, 0.0]})
    )


def test_values():
    in_trajs = []
    in_trajs.append(np.array([[1, 0], [2, 1], [3, 2], [5, 3], [7, 1]]))
    in_trajs.append(np.array([[0, 3], [2, 5], [3, 6], [6, 5]]))
    in_trajs.append(np.array([[1, 4], [3, 7], [6, 4]]))
    out_trajs = np.concatenate([np.asarray(traj) for traj in in_trajs], 0)
    py_x = np.array(out_trajs[:, 0])
    py_y = np.array(out_trajs[:, 1])
    py_cnt = []
    for traj in in_trajs:
        py_cnt.append(len(traj))
    pnt_x = cudf.Series(py_x)
    pnt_y = cudf.Series(py_y)
    cnt = cudf.Series(py_cnt)
    distance = cuspatial.directed_hausdorff_distance(pnt_x, pnt_y, cnt)

    assert_eq(
        distance,
        cudf.DataFrame(
            {
                0: [0, 3.605551, 4.472136],
                1: [4.123106, 0.0, 1.414214],
                2: [4.0, 1.414214, 0.0],
            }
        ),
    )


# def test_count_1():
# def test_count_2():
# def test_mismatched_x_y():
# def test_count_greater_than_x():
