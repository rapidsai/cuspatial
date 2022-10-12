# Copyright (c) 2020, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pytest

import cudf

import cuspatial


def test_errors():
    # t and y must have the same length
    with pytest.raises(TypeError):
        cuspatial.interpolate.CubicSpline(
            cudf.Series([0, 0, 0, 0, 0]).astype("float32"),
            cudf.Series([0]).astype("float32"),
            cudf.Series([0]).astype("int32"),
            cudf.Series([0, 1]).astype("int32"),
        )
    # length must not be zero
    with pytest.raises(ZeroDivisionError):
        cuspatial.interpolate.CubicSpline(
            cudf.Series([0, 0, 0, 0, 0]).astype("float32"),
            cudf.Series([0, 0, 0, 0, 0]).astype("float32"),
            cudf.Series([0]).astype("int32"),
            0,
        )
    # Length must be greater than 4
    with pytest.raises(ValueError):
        cuspatial.interpolate.CubicSpline(
            cudf.Series([0]).astype("float32"),
            cudf.Series([0]).astype("float32"),
            cudf.Series([0]).astype("int32"),
            1,
        )


def test_class_coefs():
    t = cudf.Series([0, 1, 2, 3, 4]).astype("float32")
    x = cudf.Series([3, 2, 3, 4, 3]).astype("float32")
    g = cuspatial.interpolate.CubicSpline(t, x)
    cudf.testing.assert_frame_equal(
        g.c,
        cudf.DataFrame(
            {
                "d3": [0.5, -0.5, -0.5, 0.5],
                "d2": [0, 3, 3, -6],
                "d1": [-1.5, -4.5, -4.5, 22.5],
                "d0": [3, 4, 4, -23],
            }
        ),
        check_dtype=False,
    )


def test_min():
    result = cuspatial.CubicSpline(
        cudf.Series([0, 1, 2, 3, 4]).astype("float32"),
        cudf.Series([3, 2, 3, 4, 3]).astype("float32"),
        cudf.Series([0, 5]).astype("int32"),
    )
    cudf.testing.assert_frame_equal(
        result.c,
        cudf.DataFrame(
            {
                "d3": [0.5, -0.5, -0.5, 0.5],
                "d2": [0, 3, 3, -6],
                "d1": [-1.5, -4.5, -4.5, 22.5],
                "d0": [3, 4, 4, -23],
            }
        ),
        check_dtype=False,
    )


def test_cusparse():
    result = cuspatial.CubicSpline(
        cudf.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).astype(
            "float32"
        ),
        cudf.Series([3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3]).astype(
            "float32"
        ),
        offset=cudf.Series([0, 5, 10, 15]).astype("int32"),
    )
    cudf.testing.assert_frame_equal(
        result.c,
        cudf.DataFrame(
            {
                "d3": [
                    0.5,
                    -0.5,
                    -0.5,
                    0.5,
                    0.5,
                    -0.5,
                    -0.5,
                    0.5,
                    0.5,
                    -0.5,
                    -0.5,
                    0.5,
                ],
                "d2": [0, 3, 3, -6, 0, 3, 3, -6, 0, 3, 3, -6],
                "d1": [
                    -1.5,
                    -4.5,
                    -4.5,
                    22.5,
                    -1.5,
                    -4.5,
                    -4.5,
                    22.5,
                    -1.5,
                    -4.5,
                    -4.5,
                    22.5,
                ],
                "d0": [3, 4, 4, -23, 3, 4, 4, -23, 3, 4, 4, -23],
            }
        ),
        check_dtype=False,
    )


def test_class_interpolation_length_five():
    t = cudf.Series([0, 1, 2, 3, 4]).astype("float32")
    x = cudf.Series([3, 2, 3, 4, 3]).astype("float32")
    g = cuspatial.interpolate.CubicSpline(t, x)
    cudf.testing.assert_series_equal(g(t), x)


def test_class_interpolation_length_six():
    t = cudf.Series([0, 1, 2, 3, 4, 5]).astype("float32")
    x = cudf.Series([3, 2, 3, 4, 3, 4]).astype("float32")
    g = cuspatial.interpolate.CubicSpline(t, x)
    cudf.testing.assert_series_equal(g(t), x)


def test_class_interpolation_length_six_splits():
    t = cudf.Series([0, 1, 2, 3, 4, 5]).astype("float32")
    x = cudf.Series([3, 2, 3, 4, 3, 4]).astype("float32")
    g = cuspatial.interpolate.CubicSpline(t, x)
    split_t = cudf.Series(np.linspace(0, 5, 11), dtype="float32")
    cudf.testing.assert_series_equal(
        g(split_t)[t * 2].reset_index(drop=True), x
    )


def test_class_triple():
    t = cudf.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).astype(
        "float32"
    )
    x = cudf.Series([3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3]).astype(
        "float32"
    )
    prefixes = cudf.Series([0, 5, 10, 15]).astype("int32")
    g = cuspatial.interpolate.CubicSpline(t, x, offset=prefixes)
    groups = cudf.Series(
        np.ravel(np.array([np.repeat(0, 5), np.repeat(1, 5), np.repeat(2, 5)]))
    )
    cudf.testing.assert_series_equal(g(t, groups=groups), x)


def test_class_triple_six():
    t = cudf.Series(
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    ).astype("float32")
    x = cudf.Series(
        [3, 2, 3, 4, 3, 1, 3, 2, 3, 4, 3, 1, 3, 2, 3, 4, 3, 1]
    ).astype("float32")
    prefixes = cudf.Series([0, 6, 12, 18]).astype("int32")
    g = cuspatial.interpolate.CubicSpline(t, x, offset=prefixes)
    groups = cudf.Series(
        np.ravel(np.array([np.repeat(0, 6), np.repeat(1, 6), np.repeat(2, 6)]))
    )
    cudf.testing.assert_series_equal(g(t, groups=groups), x)


def test_class_triple_six_splits():
    t = cudf.Series(
        [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
    ).astype("float32")
    x = cudf.Series(
        [3, 2, 3, 4, 3, 1, 3, 2, 3, 4, 3, 1, 3, 2, 3, 4, 3, 1]
    ).astype("float32")
    prefixes = cudf.Series([0, 6, 12, 18]).astype("int32")
    g = cuspatial.interpolate.CubicSpline(t, x, offset=prefixes)
    groups = cudf.Series(
        np.ravel(
            np.array([np.repeat(0, 12), np.repeat(1, 12), np.repeat(2, 12)])
        )
    )
    split_t = cudf.Series(
        np.ravel(
            (
                np.linspace(0, 5, 11),
                np.linspace(0, 5, 11),
                np.linspace(0, 5, 11),
            )
        ),
        dtype="float32",
    )
    split_t_ind = [
        0,
        2,
        4,
        6,
        8,
        10,
        11,
        13,
        15,
        17,
        19,
        21,
        22,
        24,
        26,
        28,
        30,
        32,
    ]
    cudf.testing.assert_series_equal(
        g(split_t, groups=groups)[split_t_ind].reset_index(drop=True), x
    )


def test_class_new_interpolation():
    t = cudf.Series(np.hstack((np.arange(5),) * 3)).astype("float32")
    y = cudf.Series([3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3]).astype(
        "float32"
    )
    prefix_sum = cudf.Series(cp.arange(4) * 5).astype("int32")
    new_samples = cudf.Series(np.hstack((np.linspace(0, 4, 9),) * 3)).astype(
        "float32"
    )
    curve = cuspatial.CubicSpline(t, y, offset=prefix_sum)
    new_x = cudf.Series(np.repeat(np.arange(0, 3), 9)).astype("int32")
    old_x = cudf.Series(np.repeat(np.arange(0, 3), 5)).astype("int32")
    new_points = curve(new_samples, groups=new_x)
    old_points = curve(t, groups=old_x)
    new_points_at_control_points = new_points[
        0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26
    ]
    new_points_at_control_points.index = cudf.RangeIndex(
        0, len(new_points_at_control_points)
    )
    cudf.testing.assert_series_equal(new_points_at_control_points, old_points)
