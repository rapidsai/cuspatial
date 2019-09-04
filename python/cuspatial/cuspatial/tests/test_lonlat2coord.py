# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
import cudf
from cudf.tests.utils import assert_eq
import numpy as np
import cuspatial.bindings.spatial as gis

"""
GPU accelerated coordinate transformation test: (log/lat)==>(x/y), relative to a camera origin

Note:  make sure cudf_dev conda environment is activated
"""

def test_zeros():
    coords_x, coords_y = gis.cpp_lonlat2coord(
        np.double(0.0),
        np.double(0.0),
        cudf.Series([0.0])._column,
        cudf.Series([0.0])._column
    )
    assert_eq(cudf.Series(coords_x), cudf.Series(coords_y))
    assert cudf.Series(coords_x)[0] == 0
    assert cudf.Series(coords_y)[0] == 0

def test_values():
    cam_lon= np.double(-90.66511046)
    cam_lat =np.double(42.49197018)

    py_lon=cudf.Series([-90.66518941, -90.66540743, -90.66489239])
    py_lat=cudf.Series([42.49207437, 42.49202408,42.49266787])

    #note: x/y coordinates in killometers -km 
    x,y=gis.cpp_lonlat2coord(cam_lon,cam_lat,py_lon._column, py_lat._column)
    x.data.to_array()
    y.data.to_array()
    print(cudf.Series(x))
    print(cudf.Series(y))

# def test_camera_bounds()
# def test_coord_bounds()
# def test_zero_points()
# def test_invalid_lat()
# def test_invalid_lon()

