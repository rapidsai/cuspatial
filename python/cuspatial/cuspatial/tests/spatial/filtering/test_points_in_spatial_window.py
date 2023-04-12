# Copyright (c) 2019-2023, NVIDIA CORPORATION.

import geopandas as gpd
import pytest
from geopandas.testing import assert_geoseries_equal
from shapely.geometry import Point

import cuspatial


def test_zeros():
    result = cuspatial.points_in_spatial_window(  # noqa: F841
        cuspatial.GeoSeries([Point(0, 0)]), 0, 0, 0, 0
    )
    assert result.empty


def test_centered():
    s = cuspatial.GeoSeries([Point(0, 0)])
    result = cuspatial.points_in_spatial_window(s, -1, 1, -1, 1)

    assert_geoseries_equal(result.to_geopandas(), gpd.GeoSeries([Point(0, 0)]))


@pytest.mark.parametrize(
    "coords", [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
)
def test_corners(coords):
    x, y = coords
    result = cuspatial.points_in_spatial_window(
        cuspatial.GeoSeries([Point(x, y)]), -1.1, 1.1, -1.1, 1.1
    )
    assert_geoseries_equal(result.to_geopandas(), gpd.GeoSeries([Point(x, y)]))


def test_pair():
    result = cuspatial.points_in_spatial_window(
        cuspatial.GeoSeries([Point(0, 1), Point(1, 0)]), -1.1, 1.1, -1.1, 1.1
    )
    assert_geoseries_equal(
        result.to_geopandas(), gpd.GeoSeries([Point(0, 1), Point(1, 0)])
    )


def test_oob():
    result = cuspatial.points_in_spatial_window(
        cuspatial.GeoSeries([Point(-2.0, 2.0), Point(2.0, -2.0)]),
        -1,
        1,
        -1,
        1,
    )
    assert_geoseries_equal(result.to_geopandas(), gpd.GeoSeries([]))


def test_half():
    result = cuspatial.points_in_spatial_window(
        cuspatial.GeoSeries(
            [
                Point(-1.0, 1.0),
                Point(1.0, -1.0),
                Point(3.0, 3.0),
                Point(-3.0, -3.0),
            ]
        ),
        -2,
        2,
        -2,
        2,
    )

    assert_geoseries_equal(
        result.to_geopandas(),
        gpd.GeoSeries([Point(-1.0, 1.0), Point(1.0, -1.0)]),
    )
