# Copyright (c) 2019, NVIDIA CORPORATION.

import pytest
import cudf
from cudf.tests.utils import assert_eq
import numpy as np
import cuspatial._lib.spatial as gis

"""
GPU accelerated coordinate transformation test: (log/lat)==>(x/y), relative to a camera origin

Note:  make sure cudf_dev conda environment is activated
"""

def test_camera_oob_0():
    with pytest.raises(RuntimeError):
        x, y = gis.cpp_lonlat2coord(-200, 0,
            cudf.Series([0]),
            cudf.Series([0])
        )

def test_camera_oob_1():
    with pytest.raises(RuntimeError):
        x, y = gis.cpp_lonlat2coord(200, 0,
            cudf.Series([0]),
            cudf.Series([0])
        )

def test_camera_oob_2():
    with pytest.raises(RuntimeError):
        x, y = gis.cpp_lonlat2coord(0, -100,
            cudf.Series([0]),
            cudf.Series([0])
        )

def test_camera_oob_3():
    with pytest.raises(RuntimeError):
        x, y = gis.cpp_lonlat2coord(0, 100,
            cudf.Series([0]),
            cudf.Series([0])
        )

@pytest.mark.parametrize("corner", [0, 1, 2, 3])
def test_camera_corners(corner):
    x = [-180, 180, -180, 180]
    y = [-90, 90, 90, -90]
    x, y = gis.cpp_lonlat2coord(x[corner], y[corner],
        cudf.Series(x[corner]),
        cudf.Series(y[corner])
    )
    assert x[0] == 0
    assert y[0] == 0

def test_longest_distance():
    x, y = gis.cpp_lonlat2coord(-180, -90,
        cudf.Series([180]),
        cudf.Series([90])
    )
    assert x[0] == -40000.0
    assert y[0] == -20000.0

def test_half_distance():
    x, y = gis.cpp_lonlat2coord(-180, -90,
        cudf.Series([0]),
        cudf.Series([0])
    )
    assert x[0] == -14142.135623730952
    assert y[0] == -10000.0

def test_missing_coords():
    with pytest.raises(RuntimeError):
        x, y = gis.cpp_lonlat2coord(-180, -90,
            cudf.Series(),
            cudf.Series([0])
        )

def test_zeros():
    coords_x, coords_y = gis.cpp_lonlat2coord(
        0.0,
        0.0,
        cudf.Series([0.0]),
        cudf.Series([0.0])
    )
    assert_eq(cudf.Series(coords_x), cudf.Series(coords_y))
    assert cudf.Series(coords_x)[0] == 0
    assert cudf.Series(coords_y)[0] == 0

def test_values():
    cam_lon = -90.66511046
    cam_lat = 42.49197018

    py_lon=cudf.Series([-90.66518941, -90.66540743, -90.66489239])
    py_lat=cudf.Series([42.49207437, 42.49202408,42.49266787])

    #note: x/y coordinates in killometers -km 
    x,y=gis.cpp_lonlat2coord(cam_lon, cam_lat, py_lon, py_lat)
    print(cudf.Series(x))
    print(cudf.Series(y))
