from pandas.testing import assert_series_equal
from shapely.geometry import MultiPoint, Point

import cudf

import cuspatial
from cuspatial.core.binops.equals_count import pairwise_multipoint_equals_count


def test_pairwise_multipoint_equals_count_example_1():
    p1 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    p2 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_pairwise_multipoint_equals_count_example_2():
    p1 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    p2 = cuspatial.GeoSeries([MultiPoint([Point(1, 1)])])
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([0], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_pairwise_multipoint_equals_count_example_3():
    p1 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0)]),
            MultiPoint([Point(3, 3)]),
            MultiPoint([Point(2, 2)]),
        ]
    )
    p2 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(2, 2), Point(0, 0), Point(1, 1)]),
            MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
            MultiPoint([Point(1, 1), Point(2, 2), Point(0, 0)]),
        ]
    )
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1, 0, 1], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_pairwise_multipoint_equals_count_three_match_two_mismatch():
    p1 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(3, 3)]),
            MultiPoint([Point(0, 0)]),
            MultiPoint([Point(3, 3)]),
        ]
    )
    p2 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
            MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
            MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
        ]
    )
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([0, 1, 0], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_pairwise_multipoint_equals_count_five():
    p1 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0)]),
            MultiPoint([Point(1, 1)]),
            MultiPoint([Point(2, 2)]),
            MultiPoint([Point(3, 3)]),
            MultiPoint([Point(4, 4)]),
        ]
    )
    p2 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0)]),
            MultiPoint([Point(0, 0)]),
            MultiPoint([Point(2, 2)]),
            MultiPoint([Point(2, 2)]),
            MultiPoint([Point(3, 3)]),
        ]
    )
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1, 0, 1, 0, 0], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_pairwise_multipoint_equals_two_and_three():
    p1 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0), Point(1, 1), Point(1, 1)]),
            MultiPoint([Point(0, 0), Point(1, 1), Point(1, 1)]),
        ]
    )
    p2 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
            MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
        ]
    )
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([3, 3], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_pairwise_multipoint_equals_two_and_three_one_match():
    p1 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0), Point(1, 1), Point(1, 1)]),
            MultiPoint([Point(0, 0), Point(1, 1), Point(1, 1)]),
        ]
    )
    p2 = cuspatial.GeoSeries(
        [
            MultiPoint([Point(0, 0), Point(2, 2), Point(2, 2)]),
            MultiPoint([Point(2, 2), Point(2, 2), Point(0, 0)]),
        ]
    )
    got = pairwise_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1, 1], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())
