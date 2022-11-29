from shapely.geometry import LineString

import cudf
from cudf.testing._utils import assert_eq

import cuspatial


def test_zero_pair():
    ls1 = cuspatial.GeoSeries([])
    ls2 = cuspatial.GeoSeries([])

    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = cudf.Series([], dtype="float64")

    assert_eq(got, expected)


def test_one_pair():
    ls1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (1.0, 1.0)]),
        ]
    )
    ls2 = cuspatial.GeoSeries(
        [
            LineString([(2.0, 2.0), (3.0, 3.0)]),
        ]
    )
    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = ls1.to_geopandas().distance(ls2.to_geopandas())
    assert_eq(got, expected)


def test_two_pairs():
    ls1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (1.0, 1.0), (5.0, 10.2)]),
            LineString([(7.0, 11.4), (8.0, 12.8)]),
        ]
    )
    ls2 = cuspatial.GeoSeries(
        [
            LineString([(2.0, 2.0), (3.0, 3.0)]),
            LineString(
                [(-8.0, -8.0), (-10.0, -5.0), (-13.0, -15.0), (-3.0, -6.0)]
            ),
        ]
    )

    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = ls1.to_geopandas().distance(ls2.to_geopandas())
    assert_eq(got, expected)


def test_randomized_linestring_to_linestring(linestring_generator):
    ls1 = cuspatial.GeoSeries([*linestring_generator(100, 5)])
    ls2 = cuspatial.GeoSeries([*linestring_generator(100, 5)])

    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = ls1.to_geopandas().distance(ls2.to_geopandas())
    assert_eq(got, expected)


def test_randomized_linestring_to_multilinestring(
    linestring_generator, multilinestring_generator
):
    ls1 = cuspatial.GeoSeries([*linestring_generator(100, 5)])
    ls2 = cuspatial.GeoSeries([*multilinestring_generator(100, 20, 5)])

    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = ls1.to_geopandas().distance(ls2.to_geopandas())
    assert_eq(got, expected)


def test_randomized_multilinestring_to_linestring(
    linestring_generator, multilinestring_generator
):
    ls1 = cuspatial.GeoSeries([*multilinestring_generator(100, 20, 5)])
    ls2 = cuspatial.GeoSeries([*linestring_generator(100, 5)])

    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = ls1.to_geopandas().distance(ls2.to_geopandas())
    assert_eq(got, expected)


def test_randomized_multilinestring_to_multilinestring(
    multilinestring_generator,
):
    ls1 = cuspatial.GeoSeries([*multilinestring_generator(100, 20, 5)])
    ls2 = cuspatial.GeoSeries([*multilinestring_generator(100, 20, 5)])

    got = cuspatial.pairwise_linestring_distance(ls1, ls2)
    expected = ls1.to_geopandas().distance(ls2.to_geopandas())
    assert_eq(got, expected)
