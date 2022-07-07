# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest

import cudf

import cuspatial


def test_zeros():
    result = cuspatial.points_in_spatial_window(  # noqa: F841
        0, 0, 0, 0, cudf.Series([0.0]), cudf.Series([0.0])
    )
    assert result.empty


def test_centered():
    result = cuspatial.points_in_spatial_window(
        -1, 1, -1, 1, cudf.Series([0.0]), cudf.Series([0.0])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [0.0], "y": [0.0]})
    )


@pytest.mark.parametrize(
    "coords", [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
)
def test_corners(coords):
    x, y = coords
    result = cuspatial.points_in_spatial_window(
        -1.1, 1.1, -1.1, 1.1, cudf.Series([x]), cudf.Series([y])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [x], "y": [y]})
    )


def test_pair():
    result = cuspatial.points_in_spatial_window(
        -1.1, 1.1, -1.1, 1.1, cudf.Series([0.0, 1.0]), cudf.Series([1.0, 0.0])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]})
    )


def test_oob():
    result = cuspatial.points_in_spatial_window(
        -1, 1, -1, 1, cudf.Series([-2.0, 2.0]), cudf.Series([2.0, -2.0])
    )
    cudf.testing.assert_frame_equal(result, cudf.DataFrame({"x": [], "y": []}))


def test_half():
    result = cuspatial.points_in_spatial_window(
        -2,
        2,
        -2,
        2,
        cudf.Series([-1.0, 1.0, 3.0, -3.0]),
        cudf.Series([1.0, -1.0, 3.0, -3.0]),
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [-1.0, 1.0], "y": [1.0, -1.0]})
    )
