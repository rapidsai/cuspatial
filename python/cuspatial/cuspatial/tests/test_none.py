# Copyright (c) 2022 NVIDIA CORPORATION.
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

import cuspatial


def test_got_from_geopandas():
    expected = gpd.GeoSeries([None])
    got = cuspatial.from_geopandas(expected)
    pd.testing.assert_series_equal(expected, got.to_geopandas())


def test_mix_from_geopandas():
    expected = gpd.GeoSeries([None, Point(0, 1), None])
    got = cuspatial.from_geopandas(expected)
    pd.testing.assert_series_equal(expected, got.to_geopandas())


def test_first_from_geopandas():
    expected = gpd.GeoSeries([Point(0, 1), None, None])
    got = cuspatial.from_geopandas(expected)
    pd.testing.assert_series_equal(expected, got.to_geopandas())


def test_last_from_geopandas():
    expected = gpd.GeoSeries([None, None, Point(0, 1)])
    got = cuspatial.from_geopandas(expected)
    pd.testing.assert_series_equal(expected, got.to_geopandas())


def test_middle_from_geopandas():
    expected = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
            None,
            Polygon(((-10, -2), (-10, 2), (-6, 2), (-6, -2))),
            None,
            Polygon(((-2, 8), (-2, 12), (2, 12), (2, 8))),
            Polygon(((6, 0), (8, 2), (10, 0), (8, -2))),
            None,
            Polygon(((-2, -8), (-2, -4), (2, -4), (2, -8))),
        ]
    )
    got = cuspatial.from_geopandas(expected)
    pd.testing.assert_series_equal(expected, got.to_geopandas())
