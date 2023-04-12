# Copyright (c) 2019-2023, NVIDIA CORPORATION.

import pytest
from geopandas.testing import assert_geoseries_equal
from shapely.geometry import Point

import cudf

import cuspatial


def test_camera_oob_0():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            -200, 0, cuspatial.GeoSeries([Point(0, 0)])
        )


def test_camera_oob_1():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            200, 0, cuspatial.GeoSeries([Point(0, 0)])
        )


def test_camera_oob_2():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            0, -100, cuspatial.GeoSeries([Point(0, 0)])
        )


def test_camera_oob_3():
    with pytest.raises(RuntimeError):
        result = cuspatial.sinusoidal_projection(  # noqa: F841
            0, 100, cuspatial.GeoSeries([Point(0, 0)])
        )


@pytest.mark.parametrize("corner", [0, 1, 2, 3])
def test_camera_corners(corner):
    x = [-180.0, 180.0, -180.0, 180.0]
    y = [-90.0, 90.0, 90.0, -90.0]
    lonlat = cuspatial.GeoSeries.from_points_xy([x[corner], y[corner]])
    result = cuspatial.sinusoidal_projection(x[corner], y[corner], lonlat)
    assert_geoseries_equal(
        result.to_geopandas(),
        cuspatial.GeoSeries([Point(0.0, 0.0)]).to_geopandas(),
    )


def test_longest_distance():
    result = cuspatial.sinusoidal_projection(
        -180, -90, cuspatial.GeoSeries.from_points_xy([180.0, 90.0])
    )
    assert_geoseries_equal(
        result.to_geopandas(),
        cuspatial.GeoSeries([Point(-40000.0, -20000.0)]).to_geopandas(),
    )


def test_half_distance():
    result = cuspatial.sinusoidal_projection(
        -180.0, -90.0, cuspatial.GeoSeries([Point(0.0, 0.0)])
    )
    assert_geoseries_equal(
        result.to_geopandas(),
        cuspatial.GeoSeries(
            [Point(-14142.135623730952, -10000.0)]
        ).to_geopandas(),
    )


def test_zeros():
    result = cuspatial.sinusoidal_projection(
        0.0, 0.0, cuspatial.GeoSeries([Point(0.0, 0.0)])
    )
    assert_geoseries_equal(
        result.to_geopandas(),
        cuspatial.GeoSeries([Point(0.0, 0.0)]).to_geopandas(),
    )


def test_values():
    cam_lon = -90.66511046
    cam_lat = 42.49197018

    py_lon = cudf.Series([-90.66518941, -90.66540743, -90.66489239])
    py_lat = cudf.Series([42.49207437, 42.49202408, 42.49266787])

    lonlat = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": py_lon, "y": py_lat}).interleave_columns()
    )

    # note: x/y coordinates in killometers -km
    result = cuspatial.sinusoidal_projection(cam_lon, cam_lat, lonlat)
    expected = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame(
            {
                "x": [0.0064683857, 0.024330807, -0.0178664241],
                "y": [-0.0115766660, -0.005988880, -0.0775211111],
            }
        ).interleave_columns()
    )
    assert_geoseries_equal(
        result.to_geopandas(), expected.to_geopandas(), check_less_precise=True
    )
