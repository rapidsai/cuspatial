# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf

import cuspatial
from cuspatial.utils import gis_utils


def test_missing_0():
    with pytest.raises(RuntimeError):
        cuspatial.point_in_polygon(
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_1():
    with pytest.raises(RuntimeError):
        cuspatial.point_in_polygon(
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_2():
    result = cuspatial.point_in_polygon(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series(),
        cudf.Series([0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
    )

    expected = cudf.DataFrame()
    cudf.testing.assert_frame_equal(expected, result)


def test_missing_3():
    with pytest.raises(RuntimeError):
        cuspatial.point_in_polygon(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_4():
    with pytest.raises(RuntimeError):
        cuspatial.point_in_polygon(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series(),
            cudf.Series([0.0]),
        )


def test_missing_5():
    with pytest.raises(RuntimeError):
        cuspatial.point_in_polygon(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0.0]),
            cudf.Series(),
        )


def test_zeros():
    with pytest.raises(RuntimeError):
        cuspatial.point_in_polygon(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_one_point_in():
    result = cuspatial.point_in_polygon(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([-1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1]),
    )
    expected = cudf.DataFrame({0: True})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_out():
    result = cuspatial.point_in_polygon(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([-1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1]),
    )
    expected = cudf.DataFrame({0: False})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_in_two_rings():
    result = cuspatial.point_in_polygon(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0, 4]),
        cudf.Series([-1, 0, 1, -1, -1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1, 3, 5, 3, 3]),
    )
    expected = cudf.DataFrame({0: True})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_in_two_rings_no_repeat():
    result = cuspatial.point_in_polygon(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([0, 3]),
        cudf.Series([-1, 0, 1, -1, 0, 1]),
        cudf.Series([-1, 1, -1, 3, 5, 3]),
    )
    expected = cudf.DataFrame({0: True})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_out_two_rings():
    result = cuspatial.point_in_polygon(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([0]),
        cudf.Series([0, 4]),
        cudf.Series([-1, 0, 1, -1, -1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1, 3, 5, 3, 3]),
    )
    expected = cudf.DataFrame({0: False})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_out_two_rings_no_repeat():
    result = cuspatial.point_in_polygon(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([0]),
        cudf.Series([0, 3]),
        cudf.Series([-1, 0, 1, -1, 0, 1]),
        cudf.Series([-1, 1, -1, 3, 5, 3]),
    )
    expected = cudf.DataFrame({0: False})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_in_one_out_two_rings():
    result = cuspatial.point_in_polygon(
        cudf.Series([0, 1]),
        cudf.Series([0, 1]),
        cudf.Series([0]),
        cudf.Series([0, 4]),
        cudf.Series([-1, 0, 1, -1, -1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1, 3, 5, 3, 3]),
    )
    expected = cudf.DataFrame({0: [True, False]})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_out_one_in_two_rings():
    result = cuspatial.point_in_polygon(
        cudf.Series([1, 0]),
        cudf.Series([1, 0]),
        cudf.Series([0]),
        cudf.Series([0, 4]),
        cudf.Series([-1, 0, 1, -1, -1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1, 3, 5, 3, 3]),
    )
    expected = cudf.DataFrame({0: [False, True]})
    cudf.testing.assert_frame_equal(expected, result)


def test_two_points_out_two_rings():
    result = cuspatial.point_in_polygon(
        cudf.Series([1, -1]),
        cudf.Series([1, 1]),
        cudf.Series([0]),
        cudf.Series([0, 4]),
        cudf.Series([-1, 0, 1, -1, -1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1, 3, 5, 3, 3]),
    )
    expected = cudf.DataFrame({0: [False, False]})
    cudf.testing.assert_frame_equal(expected, result)


def test_two_points_in_two_rings():
    result = cuspatial.point_in_polygon(
        cudf.Series([0, 0]),
        cudf.Series([0, 4]),
        cudf.Series([0]),
        cudf.Series([0, 4]),
        cudf.Series([-1, 0, 1, -1, -1, 0, 1, -1]),
        cudf.Series([-1, 1, -1, -1, 3, 5, 3, 3]),
    )
    expected = cudf.DataFrame({0: [True, True]})
    cudf.testing.assert_frame_equal(expected, result)


def test_three_points_two_features():
    result = cuspatial.point_in_polygon(
        cudf.Series([0, -8, 6.0]),
        cudf.Series([0, -8, 6.0]),
        cudf.Series([0, 1]),
        cudf.Series([0, 5]),
        cudf.Series([-10.0, 5, 5, -10, -10, 0, 10, 10, 0, 0]),
        cudf.Series([-10.0, -10, 5, 5, -10, 0, 0, 10, 10, 0]),
    )
    expected = cudf.DataFrame()
    expected[0] = [True, True, False]
    expected[1] = [False, False, True]
    cudf.testing.assert_frame_equal(expected, result)


def test_pip_bitmap_column_to_binary_array():
    col = cudf.Series([0, 13, 3, 9])._column
    got = gis_utils.pip_bitmap_column_to_binary_array(col, width=4)
    expected = np.array(
        [[0, 0, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]], dtype="int8"
    )
    np.testing.assert_array_equal(got.copy_to_host(), expected)

    col = cudf.Series([])._column
    got = gis_utils.pip_bitmap_column_to_binary_array(col, width=0)
    expected = np.array([], dtype="int8").reshape(0, 0)
    np.testing.assert_array_equal(got.copy_to_host(), expected)

    col = cudf.Series([None, None], dtype="float64")._column
    got = gis_utils.pip_bitmap_column_to_binary_array(col, width=0)
    expected = np.array([], dtype="int8").reshape(2, 0)
    np.testing.assert_array_equal(got.copy_to_host(), expected)

    col = cudf.Series([238, 13, 29594])._column
    got = gis_utils.pip_bitmap_column_to_binary_array(col, width=15)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
        ],
        dtype="int8",
    )
    np.testing.assert_array_equal(got.copy_to_host(), expected)

    col = cudf.Series([0, 0, 0])._column
    got = gis_utils.pip_bitmap_column_to_binary_array(col, width=3)
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype="int8")
    np.testing.assert_array_equal(got.copy_to_host(), expected)
