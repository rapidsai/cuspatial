# Copyright (c) 2023, NVIDIA CORPORATION

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

import cuspatial


def test_manual_polygons():
    gpdlhs = gpd.GeoSeries([Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8)))] * 6)
    gpdrhs = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
            Polygon(((-10, -2), (-10, 2), (-6, 2), (-6, -2))),
            Polygon(((-2, 8), (-2, 12), (2, 12), (2, 8))),
            Polygon(((6, 0), (8, 2), (10, 0), (8, -2))),
            Polygon(((-2, -8), (-2, -4), (2, -4), (2, -8))),
        ]
    )
    rhs = cuspatial.from_geopandas(gpdrhs)
    lhs = cuspatial.from_geopandas(gpdlhs)
    got = lhs.contains(rhs).values_host
    expected = gpdlhs.contains(gpdrhs).values
    assert (got == expected).all()
    got = rhs.contains(lhs).values_host
    expected = gpdrhs.contains(gpdlhs).values
    assert (got == expected).all()


def test_same():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = lhs.contains(rhs)
    expected = gpdlhs.contains(gpdrhs)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_adjacent():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])])
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = lhs.contains(rhs)
    expected = gpdlhs.contains(gpdrhs)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_interior():
    lhs = cuspatial.GeoSeries(
        [Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])]
    )
    rhs = cuspatial.GeoSeries(
        [Polygon([(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)])]
    )
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = lhs.contains(rhs)
    expected = gpdlhs.contains(gpdrhs)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.parametrize(
    "object",
    [
        Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
        MultiPolygon(
            [
                Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
                Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
            ]
        ),
    ],
)
def test_self_contains(object):
    gpdobject = gpd.GeoSeries(object)
    object = cuspatial.from_geopandas(gpdobject)
    got = object.contains(object).values_host
    expected = gpdobject.contains(gpdobject).values
    assert (got == expected).all()


def test_complex_input():
    gpdobject = gpd.GeoSeries(
        [
            Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
            Polygon(
                ([0, 0], [1, 1], [1, 0], [0, 0]),
                [([0, 0], [1, 1], [1, 0], [0, 0])],
            ),
            MultiPolygon(
                [
                    Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
                    Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]),
                    Polygon(
                        ([0, 0], [1, 1], [1, 0], [0, 0]),
                        [([0, 0], [1, 1], [1, 0], [0, 0])],
                    ),
                ]
            ),
        ]
    )
    object = cuspatial.from_geopandas(gpdobject)
    got = object.contains(object).values_host
    expected = gpdobject.contains(gpdobject).values
    assert (got == expected).all()
