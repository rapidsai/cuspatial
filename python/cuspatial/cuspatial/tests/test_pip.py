# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
import cudf
from cudf.tests.utils import assert_eq
import numpy as np
import cuspatial.bindings.spatial as gis

def test_zeros():
    result = gis.cpp_point_in_polygon_bitmap(
        cudf.Series([0.0])._column,
        cudf.Series([0.0])._column,
        cudf.Series([0.0])._column,
        cudf.Series([0.0])._column,
        cudf.Series([0.0])._column,
        cudf.Series([0.0])._column
    )
    assert_eq(cudf.Series(result), cudf.Series([0]).astype('int32'))

def test_dataset():
    points_x = cudf.Series([0, -8, 6.0])
    points_y = cudf.Series([0, -8, 6.0])
    polygon_front_points = cudf.Series([1, 2]).astype('int32')
    polygon_rear_points = cudf.Series([5, 10]).astype('int32')
    polygon_x = cudf.Series([-10.0, 5, 5, -10, -10, 0, 10, 10, 0, 0])
    polygon_y = cudf.Series([-10.0, -10, 5, 5, -10, 0, 0, 10, 10, 0])
    result = gis.cpp_point_in_polygon_bitmap(
        points_x._column,
        points_y._column,
        polygon_front_points._column,
        polygon_rear_points._column,
        polygon_x._column,
        polygon_y._column
    )
    # The result of cpp_point_in_polygon_bitmap is a binary bitmap of
    # coordinates inside of the polgyon. Why does it return 3 values?
    print(np.binary_repr(result.data.to_array()[0], width=2))
    assert_eq(cudf.Series(result), cudf.Series([3, 1, 2]).astype('int32'))

