import pandas as pd
from pandas.testing import assert_series_equal
from shapely.geometry import LineString, Point, Polygon

import cudf
import cuspatial


def test_basic_contains_none_outside():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(2, 2)])
    pd.testing.assert_series_equal(
        lhs._basic_contains_none(rhs).to_pandas(),
        pd.Series([True])
    )

def test_basic_contains_none_inside():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0.5, 0.5)])
    pd.testing.assert_series_equal(
        lhs._basic_contains_none(rhs).to_pandas(),
        pd.Series([False])
    )

def test_basic_contains_none_point():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0, 0)])
    pd.testing.assert_series_equal(
        lhs._basic_contains_none(rhs).to_pandas(),
        pd.Series([False])
    )

def test_basic_contains_none_edge():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0, 0.5)])
    pd.testing.assert_series_equal(
        lhs._basic_contains_none(rhs).to_pandas(),
        pd.Series([False])
    )
