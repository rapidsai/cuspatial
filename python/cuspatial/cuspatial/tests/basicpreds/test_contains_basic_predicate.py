# Copyright (c) 2023, NVIDIA CORPORATION.

from shapely.geometry import LineString, Point, Polygon

import cuspatial
from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_any,
    _basic_contains_count,
)


def test_basic_contains_any_outside():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(2, 2)])
    got = _basic_contains_any(lhs, rhs).to_pandas()
    expected = [False]
    assert (got == expected).all()


def test_basic_contains_any_inside():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([LineString([(0.5, 0.5), (1.5, 1.5)])])
    got = _basic_contains_any(lhs, rhs).to_pandas()
    expected = [True]
    assert (got == expected).all()


def test_basic_contains_any_point():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0, 0)])
    got = _basic_contains_any(lhs, rhs).to_pandas()
    expected = [True]
    assert (got == expected).all()


def test_basic_contains_any_edge():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0, 0.5)])
    got = _basic_contains_any(lhs, rhs).to_pandas()
    expected = [True]
    assert (got == expected).all()


def test_basic_contains_count_outside():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(2, 2)])
    got = _basic_contains_count(lhs, rhs).to_pandas()
    expected = [0]
    assert (got == expected).all()


def test_basic_contains_count_inside():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([LineString([(0.5, 0.5), (1.5, 1.5)])])
    got = _basic_contains_count(lhs, rhs).to_pandas()
    expected = [1]
    assert (got == expected).all()


def test_basic_contains_count_point():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0, 0)])
    got = _basic_contains_count(lhs, rhs).to_pandas()
    expected = [0]
    assert (got == expected).all()


def test_basic_contains_count_edge():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Point(0, 0.5)])
    got = _basic_contains_count(lhs, rhs).to_pandas()
    expected = [0]
    assert (got == expected).all()
