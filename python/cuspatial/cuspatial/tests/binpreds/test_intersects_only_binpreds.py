# Copyright (c) 2020-2023, NVIDIA CORPORATION

import pandas as pd
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cuspatial


def test_point_intersects_point():
    g1 = cuspatial.GeoSeries([Point(0.0, 0.0)])
    g2 = cuspatial.GeoSeries([Point(0.0, 0.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_points_intersects_points():
    g1 = cuspatial.GeoSeries([Point(0.0, 0.0), Point(0.0, 0.0)])
    g2 = cuspatial.GeoSeries([Point(0.0, 0.0), Point(0.0, 1.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_point():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (0.0, 1.0)])])
    g2 = cuspatial.GeoSeries([Point(0.0, 0.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_extra_linestring_intersects_point():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([Point(1.0, 1.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_extra_linestring_intersects_point_2():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([Point(0.5, 0.5)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestrings_intersects_points():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries([Point(0.0, 0.0), Point(0.0, 0.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestrings_intersects_points_only_one():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries([Point(0.0, 0.0), Point(2.0, 2.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestrings_intersects_points_only_one_reversed():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries([Point(2.0, 2.0), Point(0.0, 0.0)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_three_linestrings_intersects_three_points_match_middle():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(2.0, 2.0),
            Point(0.0, 0.0),
            Point(2.0, 2.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_three_linestrings_intersects_three_points_exclude_middle():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(2.0, 2.0),
            Point(0.0, 0.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_linestrings_intersects_100_points(
    linestring_generator, point_generator
):
    g1 = cuspatial.GeoSeries([*linestring_generator(100, 4)])
    g2 = cuspatial.GeoSeries([*point_generator(100)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multipoint():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([MultiPoint([(0.0, 0.0), (1.0, 1.0)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multipoint_midpoint():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([MultiPoint([(0.5, 0.5), (0.5, 0.5)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multipoint_midpoint_disordered():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([MultiPoint([(0.0, 0.0), (0.5, 0.5)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multipoint_midpoint_disordered_2():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([MultiPoint([(0.5, 0.5), (0.0, 0.0)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_three_linestrings_intersects_middle_multipoint():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            MultiPoint([(2.0, 2.0), (3.0, 3.0)]),
            MultiPoint([(0.0, 0.0), (0.0, 0.0)]),
            MultiPoint([(2.0, 2.0), (4.0, 4.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_three_linestrings_intersects_not_middle_multipoint():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            MultiPoint([(0.0, 0.0), (0.0, 0.0)]),
            MultiPoint([(2.0, 2.0), (3.0, 3.0)]),
            MultiPoint([(0.0, 0.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multipoint_cross_intersection():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries(
        [MultiPoint([(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)])]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.xfail(reason="Multipoints not supported yet.")
def test_linestring_intersects_multipoint_implicit_cross_intersection():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([MultiPoint([(0.0, 1.0), (1.0, 0.0)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.xfail(reason="Multipoints not supported yet.")
def test_100_linestrings_intersects_100_multipoints(
    linestring_generator, multipoint_generator
):
    g1 = cuspatial.GeoSeries([*linestring_generator(15, 4)])
    g2 = cuspatial.GeoSeries([*multipoint_generator(15, 4)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_linestring_crosses():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([LineString([(0.0, 1.0), (1.0, 0.0)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestering_intersects_linestring_parallel():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (0.0, 1.0)])])
    g2 = cuspatial.GeoSeries([LineString([(1.0, 0.0), (1.0, 1.0)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_linestring_overlaps():
    g1 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (1.0, 1.0)])])
    g2 = cuspatial.GeoSeries([LineString([(0.0, 0.0), (0.5, 0.5)])])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_two_linestrings_intersects_two_linestrings_parallel():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(1.0, 1.0), (1.0, 2.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(2.0, 0.0), (2.0, 2.0)]),
            LineString([(1.0, 0.0), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_two_linestrings_intersects_two_linestrings_overlaps():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.5, 0.5)]),
            LineString([(0.5, 0.5), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_two_linestrings_intersects_two_linestrings_touches():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (1.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(1.0, 1.0), (2.0, 2.0)]),
            LineString([(1.0, 0.0), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_two_linestrings_intersects_two_linestrings_single():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 1.0), (1.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_two_linestrings_intersects_two_linestrings_single_reversed():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_three_linestrings_intersects_three_linestrings_middle():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_three_linestrings_intersects_three_linestrings_not_middle():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 1.0), (1.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_linestrings_intersects_100_linestrings(linestring_generator):
    g1 = cuspatial.GeoSeries([*linestring_generator(100, 5)])
    g2 = cuspatial.GeoSeries([*linestring_generator(100, 5)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_linestrings_intersects_100_multilinestrings(
    linestring_generator, multilinestring_generator
):
    g1 = cuspatial.GeoSeries([*linestring_generator(100, 5)])
    g2 = cuspatial.GeoSeries([*multilinestring_generator(100, 5, 5)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_multilinestrings_intersects_100_linestrings(
    linestring_generator, multilinestring_generator
):
    g1 = cuspatial.GeoSeries([*multilinestring_generator(100, 5, 5)])
    g2 = cuspatial.GeoSeries([*linestring_generator(100, 5)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_multilinestrings_intersects_100_multilinestrings(
    multilinestring_generator,
):
    g1 = cuspatial.GeoSeries([*multilinestring_generator(100, 5, 5)])
    g2 = cuspatial.GeoSeries([*multilinestring_generator(100, 5, 5)])
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multilinestring():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            MultiLineString(
                [
                    [(0.0, 0.0), (0.0, 1.0)],
                    [(0.0, 1.0), (1.0, 1.0)],
                ]
            ),
            MultiLineString(
                [
                    [(0.0, 0.0), (0.0, 1.0)],
                    [(0.0, 1.0), (1.0, 1.0)],
                ]
            ),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_multilinestring_intersects_linestring():
    g1 = cuspatial.GeoSeries(
        [
            MultiLineString(
                [
                    [(0.0, 0.0), (0.0, 1.0)],
                    [(0.0, 1.0), (1.0, 1.0)],
                ]
            ),
            MultiLineString(
                [
                    [(0.0, 0.0), (0.0, 1.0)],
                    [(0.0, 1.0), (1.0, 1.0)],
                ]
            ),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_polygon():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_polygon_intersects_linestring():
    g1 = cuspatial.GeoSeries(
        [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_multipolygon_intersects_linestring():
    g1 = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_intersects_multipolygon():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_polygon_intersects_multipolygon():
    g1 = cuspatial.GeoSeries(
        [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_multipolygon_intersects_polygon():
    g1 = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_multipolygon_intersects_multipolygon():
    g1 = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                    Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
                ]
            ),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.intersects(g2)
    expected = gpdg1.intersects(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_point_disjoint_linestring():
    g1 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.disjoint(g2)
    expected = gpdg1.disjoint(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_disjoint_point():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.disjoint(g2)
    expected = gpdg1.disjoint(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_disjoint_linestring():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.disjoint(g2)
    expected = gpdg1.disjoint(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_contains_point():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.contains_properly(g2)
    expected = gpdg1.contains(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_covers_point():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.covers(g2)
    expected = gpdg1.covers(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_crosses_point():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.crosses(g2)
    expected = gpdg1.crosses(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def linestring_crosses_linestring():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.crosses(g2)
    expected = gpdg1.crosses(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def linestring_crosses_polygon():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.crosses(g2)
    expected = gpdg1.crosses(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_overlaps_point():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Point(0.0, 0.0),
            Point(0.0, 0.0),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.overlaps(g2)
    expected = gpdg1.overlaps(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_overlaps_linestring():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 1.0), (1.0, 1.0)]),
            LineString([(0.0, 1.0), (1.0, 1.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.overlaps(g2)
    expected = gpdg1.overlaps(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_overlaps_polygon():
    g1 = cuspatial.GeoSeries(
        [
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            LineString([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    g2 = cuspatial.GeoSeries(
        [
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
            Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]),
        ]
    )
    gpdg1 = g1.to_geopandas()
    gpdg2 = g2.to_geopandas()
    got = g1.overlaps(g2)
    expected = gpdg1.overlaps(gpdg2)
    pd.testing.assert_series_equal(expected, got.to_pandas())
