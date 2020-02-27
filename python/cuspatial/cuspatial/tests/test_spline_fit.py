# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf
from cudf.tests.utils import assert_eq

import cuspatial

import pytest


@pytest.mark.xfail(raises=TypeError, reason="size must be an integer")
def test_bad_size_type():
    # error handling
    result = cuspatial.interpolate.CubicSpline(
        cudf.Series([0]).astype('float32'), cudf.Series([0]).astype('float32'),
        cudf.Series([0]).astype('int32'), cudf.Series([0, 1]).astype('int32')
    )
    assert_eq(result, cudf.DataFrame([0, 0, 0, 0]))


@pytest.mark.xfail(raises=ZeroDivisionError, reason="modulo by zero")
def test_size_is_zero():
    # error handling
    result = cuspatial.interpolate.CubicSpline(
        cudf.Series([0]).astype('float32'), cudf.Series([0]).astype('float32'),
        cudf.Series([0]).astype('int32'), 0
    )
    assert_eq(result, cudf.DataFrame([0, 0, 0, 0]))

@pytest.mark.xfail(raises=ValueError, reason="splines of length")
def test_zeros():
    result = cuspatial.interpolate.CubicSpline(
        cudf.Series([0]).astype('float32'), cudf.Series([0]).astype('float32'),
        cudf.Series([0]).astype('int32'), 1
    )
    print(result.c)
    assert_eq(result.c, cudf.DataFrame({"d3": 0, "d2": 0, "d1": 0, "d0": 0}).astype('float32'))


def test_class():
    t = cudf.Series([0, 1, 2, 3, 4])
    x = cudf.Series([3, 2, 3, 4, 3])
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
    result = cuspatial.cubic_spline_2(
        cudf.Series([0, 1, 2, 3, 4]).astype("float32"),
        cudf.Series([3, 2, 3, 4, 3]).astype("float32"),
        cudf.Series([0, 0]).astype("int32"),
        cudf.Series([0, 5]).astype("int32"),
    )
    print(result)
    assert_eq(
        result,
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
    result = cuspatial.cubic_spline_2(
        cudf.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).astype(
            "float32"
        ),
        cudf.Series([3, 2, 3, 4, 3, 3, 2, 3, 4, 3, 3, 2, 3, 4, 3]).astype(
            "float32"
        ),
        cudf.Series([0, 0, 1, 2]).astype("int32"),
        cudf.Series([0, 5, 10, 15]).astype("int32"),
    )
    assert_eq(
        result,
        cudf.DataFrame(
            {
                "d3": [
                    0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
                ],
                "d2": [
                    0, 3, 3, -6, 0, 3, 3, -6, 0, 3, 3, -6,
                ],
                "d1": [
                    -1.5, -4.5, -4.5, 22.5, -1.5, -4.5, -4.5, 22.5, -1.5, -4.5, -4.5, 22.5
                ],
                "d0": [
                    3, 4, 4, -23, 3, 4, 4, -23, 3, 4, 4, -23
                ],
            }
        ),
        check_dtype=False,
    )
