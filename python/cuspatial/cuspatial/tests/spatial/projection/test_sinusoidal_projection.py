# Copyright (c) 2019-2022, NVIDIA CORPORATION.

import pytest

import cudf

import cuspatial


def test_camera_oob_0():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            -200, 0, cudf.Series([0]), cudf.Series([0])
        )


def test_camera_oob_1():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            200, 0, cudf.Series([0]), cudf.Series([0])
        )


def test_camera_oob_2():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            0, -100, cudf.Series([0]), cudf.Series([0])
        )


def test_camera_oob_3():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            0, 100, cudf.Series([0]), cudf.Series([0])
        )


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
def test_camera_corners(corner):
    x = [-180.0, 180.0, -180.0, 180.0]
    y = [-90.0, 90.0, 90.0, -90.0]
    result = cuspatial.sinusoidal_projection(
        x[corner], y[corner], cudf.Series(x[corner]), cudf.Series(y[corner])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [0.0], "y": [0.0]})
    )


def test_longest_distance():
    result = cuspatial.sinusoidal_projection(
        -180, -90, cudf.Series([180.0]), cudf.Series([90.0])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [-40000.0], "y": [-20000.0]})
    )


def test_half_distance():
    result = cuspatial.sinusoidal_projection(
        -180.0, -90.0, cudf.Series([0.0]), cudf.Series([0.0])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [-14142.135623730952], "y": [-10000.0]})
    )


def test_missing_coords():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            -180.0, -90.0, cudf.Series(), cudf.Series([0.0])
        )


def test_zeros():
    result = cuspatial.sinusoidal_projection(
        0.0, 0.0, cudf.Series([0.0]), cudf.Series([0.0])
    )
    cudf.testing.assert_frame_equal(
        result, cudf.DataFrame({"x": [0.0], "y": [0.0]})
    )


def test_values():
    cam_lon = -90.66511046
    cam_lat = 42.49197018

    py_lon = cudf.Series([-90.66518941, -90.66540743, -90.66489239])
    py_lat = cudf.Series([42.49207437, 42.49202408, 42.49266787])

    # note: x/y coordinates in killometers -km
    result = cuspatial.sinusoidal_projection(cam_lon, cam_lat, py_lon, py_lat)
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x": [0.0064683857, 0.024330807, -0.0178664241],
                "y": [-0.0115766660, -0.005988880, -0.0775211111],
            }
        ),
    )
