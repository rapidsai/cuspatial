from pandas.testing import assert_series_equal
from shapely.geometry import MultiPoint, Point

import cudf

import cuspatial
from cuspatial.core.binops.equals_count import allpairs_multipoint_equals_count


def test_allpairs_multipoint_equals_count_one_one_hit():
    p1 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    p2 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    got = allpairs_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_allpairs_multipoint_equals_count_one_one_miss():
    p1 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    p2 = cuspatial.GeoSeries([MultiPoint([Point(1, 1)])])
    got = allpairs_multipoint_equals_count(p1, p2)
    expected = cudf.Series([0], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_allpairs_multipoint_equals_count_three_three_one_mismatch():
    p1 = cuspatial.GeoSeries(
        [MultiPoint([Point(0, 0), Point(3, 3), Point(2, 2)])]
    )
    p2 = cuspatial.GeoSeries(
        [MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)])]
    )
    got = allpairs_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1, 0, 1], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_allpairs_multipoint_equals_count_three_match_two_mismatch():
    p1 = cuspatial.GeoSeries(
        [MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)])]
    )
    p2 = cuspatial.GeoSeries(
        [MultiPoint([Point(3, 3), Point(1, 1), Point(3, 3)])]
    )
    got = allpairs_multipoint_equals_count(p1, p2)
    expected = cudf.Series([0, 1, 0], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())


def test_allpairs_multipoint_equals_count_five():
    p1 = cuspatial.GeoSeries(
        [
            MultiPoint(
                [
                    Point(0, 0),
                    Point(1, 1),
                    Point(2, 2),
                    Point(3, 3),
                    Point(4, 4),
                ]
            )
        ]
    )
    p2 = cuspatial.GeoSeries(
        [
            MultiPoint(
                [
                    Point(0, 0),
                    Point(0, 0),
                    Point(2, 2),
                    Point(2, 2),
                    Point(3, 3),
                ]
            )
        ]
    )
    got = allpairs_multipoint_equals_count(p1, p2)
    expected = cudf.Series([2, 0, 2, 1, 0], dtype="uint32")
    assert_series_equal(got.to_pandas(), expected.to_pandas())
