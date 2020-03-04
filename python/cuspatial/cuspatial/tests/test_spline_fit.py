# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


def test_errors():
    # t and y must have the same length
    with pytest.raises(TypeError):
        result = cuspatial.interpolate.CubicSpline(
            cudf.Series([0, 0, 0, 0, 0]).astype("float32"),
            cudf.Series([0]).astype("float32"),
            cudf.Series([0]).astype("int32"),
            cudf.Series([0, 1]).astype("int32"),
        )
    # length must not be zero
    with pytest.raises(ZeroDivisionError):
        result = cuspatial.interpolate.CubicSpline(
            cudf.Series([0, 0, 0, 0, 0]).astype("float32"),
            cudf.Series([0, 0, 0, 0, 0]).astype("float32"),
            cudf.Series([0]).astype("int32"),
            0,
        )
    # Length must be greater than 4
    with pytest.raises(ValueError):
        result = cuspatial.interpolate.CubicSpline(
            cudf.Series([0]).astype("float32"),
            cudf.Series([0]).astype("float32"),
            cudf.Series([0]).astype("int32"),
            1,
        )


def test_class_coefs():
    t = cudf.Series([0, 1, 2, 3, 4]).astype("float32")
    x = cudf.Series([3, 2, 3, 4, 3]).astype("float32")
    g = cuspatial.interpolate.CubicSpline(t, x)
    assert_eq(
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
    assert_eq(
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
        prefixes=cudf.Series([0, 5, 10, 15]).astype("int32"),
    )
    assert_eq(
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


def test_class_interpolation():
    t = cudf.Series([0, 1, 2, 3, 4]).astype("float32")
    x = cudf.Series([3, 2, 3, 4, 3]).astype("float32")
    g = cuspatial.interpolate.CubicSpline(t, x)
    assert_eq(g(t), x)


def test_class_triple():
    t = cudf.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).astype(
        "float32"
    )
    x = cudf.Series([3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3]).astype(
        "float32"
    )
    g = cuspatial.interpolate.CubicSpline(
        t, x, prefixes=cudf.Series([0, 5, 10, 15]).astype("int32")
    )
    groups = np.array([np.repeat(0, 5), np.repeat(1, 5), np.repeat(2, 5)])
    assert_eq(g(t, groups=cudf.Series(groups)), x)
