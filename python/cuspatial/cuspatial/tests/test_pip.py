# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial
from cuspatial.utils import gis_utils


def test_missing_0():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(  # noqa: F841
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_1():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(  # noqa: F841
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_2():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(  # noqa: F841
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_3():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(  # noqa: F841
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
        )


def test_missing_4():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(  # noqa: F841
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
        )


def test_missing_5():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(  # noqa: F841
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
        )


def test_zeros():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
    )
    expected = cudf.DataFrame({"in_polygon_0.0": False})
    assert_eq(result, expected)


def test_one_point_in():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([1]),
        cudf.Series([3]),
        cudf.Series([-1, 0, 1]),
        cudf.Series([-1, 1, -1]),
    )
    expected = cudf.DataFrame({"in_polygon_1": True})
    assert_eq(result, expected)


def test_one_point_out():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([3]),
        cudf.Series([-1, 0, 1]),
        cudf.Series([-1, 1, -1]),
    )
    expected = cudf.DataFrame({"in_polygon_1": False})
    assert_eq(result, expected)


def test_dataset():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([0, -8, 6.0]),
        cudf.Series([0, -8, 6.0]),
        cudf.Series([1, 2]),
        cudf.Series([5, 10]),
        cudf.Series([-10.0, 5, 5, -10, -10, 0, 10, 10, 0, 0]),
        cudf.Series([-10.0, -10, 5, 5, -10, 0, 0, 10, 10, 0]),
    )
    expected = cudf.DataFrame()
    expected["in_polygon_1"] = [True, True, False]
    expected["in_polygon_2"] = [True, False, True]
    assert_eq(result, expected)


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

    col = cudf.Series([None, None])._column
    got = gis_utils.pip_bitmap_column_to_binary_array(col, width=0)
    expected = np.array([], dtype="int8").reshape(2, 0)
    breakpoint()
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
