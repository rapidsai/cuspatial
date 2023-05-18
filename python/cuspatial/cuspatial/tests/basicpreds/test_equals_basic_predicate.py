import pandas as pd
from pandas.testing import assert_series_equal
from shapely.geometry import Point

import cuspatial
from cuspatial.core.binpreds.basic_predicates import _basic_equals_any


def test_single_true():
    p1 = cuspatial.GeoSeries([Point(0, 0)])
    p2 = cuspatial.GeoSeries([Point(0, 0)])
    result = _basic_equals_any(p1, p2)
    assert_series_equal(result.to_pandas(), pd.Series([True]))


def test_single_false():
    p1 = cuspatial.GeoSeries([Point(0, 0)])
    p2 = cuspatial.GeoSeries([Point(1, 1)])
    result = _basic_equals_any(p1, p2)
    assert_series_equal(result.to_pandas(), pd.Series([False]))


def test_true_false():
    p1 = cuspatial.GeoSeries([Point(0, 0), Point(1, 1)])
    p2 = cuspatial.GeoSeries([Point(0, 0), Point(2, 2)])
    result = _basic_equals_any(p1, p2)
    assert_series_equal(result.to_pandas(), pd.Series([True, False]))


def test_false_true():
    p1 = cuspatial.GeoSeries([Point(0, 0), Point(0, 0)])
    p2 = cuspatial.GeoSeries([Point(1, 1), Point(0, 0)])
    result = _basic_equals_any(p1, p2)
    assert_series_equal(result.to_pandas(), pd.Series([False, True]))


def test_true_false_true():
    p1 = cuspatial.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
    p2 = cuspatial.GeoSeries([Point(0, 0), Point(2, 2), Point(2, 2)])
    result = _basic_equals_any(p1, p2)
    assert_series_equal(result.to_pandas(), pd.Series([True, False, True]))


def test_false_true_false():
    p1 = cuspatial.GeoSeries([Point(0, 0), Point(0, 0), Point(0, 0)])
    p2 = cuspatial.GeoSeries([Point(1, 1), Point(0, 0), Point(2, 2)])
    result = _basic_equals_any(p1, p2)
    assert_series_equal(result.to_pandas(), pd.Series([False, True, False]))
