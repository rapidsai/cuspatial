from pandas.testing import assert_series_equal
from shapely.geometry import Point

import cudf

import cuspatial
from cuspatial.core.binops.equals_count import allpairs_multipoint_equals_count


def test_allpairs_multipoint_equals_count():
    p1 = cuspatial.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
    p2 = cuspatial.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
    got = allpairs_multipoint_equals_count(p1, p2)
    expected = cudf.Series([1, 1, 1])
    assert_series_equal(got, expected)
