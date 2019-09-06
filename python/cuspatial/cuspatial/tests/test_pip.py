# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
import cudf
from cudf.tests.utils import assert_eq
import numpy as np
import cuspatial

def test_missing_0():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0])
        )

def test_missing_1():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0])
        )

def test_missing_2():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0])
        )

def test_missing_3():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0]),
            cudf.Series([0.0])
        )

def test_missing_4():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series(),
            cudf.Series([0.0])
        )

def test_missing_5():
    with pytest.raises(RuntimeError):
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series([0.0]),
            cudf.Series()
        )

def test_zeros():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0])
    )
    assert_eq(cudf.Series(result), cudf.Series([0]).astype('int32'))

def test_one_point_in():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([0]),
        cudf.Series([0]),
        cudf.Series([1]),
        cudf.Series([3]),
        cudf.Series([-1, 0, 1]),
        cudf.Series([-1, 1, -1])
    )
    assert_eq(cudf.Series(result), cudf.Series([1]).astype('int32'))

def test_one_point_out():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([1]),
        cudf.Series([3]),
        cudf.Series([-1, 0, 1]),
        cudf.Series([-1, 1, -1])
    )
    assert_eq(cudf.Series(result), cudf.Series([0]).astype('int32'))

def test_dataset():
    result = cuspatial.point_in_polygon_bitmap(
        cudf.Series([0, -8, 6.0]),
        cudf.Series([0, -8, 6.0]),
        cudf.Series([1, 2]),
        cudf.Series([5, 10]),
        cudf.Series([-10.0, 5, 5, -10, -10, 0, 10, 10, 0, 0]),
        cudf.Series([-10.0, -10, 5, 5, -10, 0, 0, 10, 10, 0]),
    )
    # The result of point_in_polygon_bitmap is a binary bitmap of
    # coordinates inside of the polgyon.
    print(np.binary_repr(result.data.to_array()[0], width=2))
    assert_eq(cudf.Series(result), cudf.Series([3, 1, 2]).astype('int32'))

