# Copyright (c) 2023, NVIDIA CORPORATION

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

import cuspatial


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
