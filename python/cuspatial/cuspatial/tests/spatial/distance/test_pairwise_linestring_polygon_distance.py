# Copyright (c) 2023, NVIDIA CORPORATION.

import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

import cudf
from cudf.testing import assert_series_equal

import cuspatial


def test_linestring_polygon_empty():
    lhs = cuspatial.GeoSeries.from_linestrings_xy([], [0], [0])
    rhs = cuspatial.GeoSeries.from_polygons_xy([], [0], [0], [0])

    got = cuspatial.pairwise_linestring_polygon_distance(lhs, rhs)

    expect = cudf.Series([], dtype="f8")

    assert_series_equal(got, expect)


@pytest.mark.parametrize(
    "linestrings",
    [
        [LineString([(0, 0), (1, 1)])],
        [MultiLineString([[(1, 1), (2, 2)], [(10, 10), (11, 11)]])],
    ],
)
@pytest.mark.parametrize(
    "polygons",
    [
        [Polygon([(0, 1), (1, 0), (-1, 0), (0, 1)])],
        [
            MultiPolygon(
                [
                    Polygon([(-2, 0), (-1, 0), (-1, -1), (-2, 0)]),
                    Polygon([(1, 0), (2, 0), (1, -1), (1, 0)]),
                ]
            )
        ],
    ],
)
def test_one_pair(linestrings, polygons):
    lhs = gpd.GeoSeries(linestrings)
    rhs = gpd.GeoSeries(polygons)

    dlhs = cuspatial.GeoSeries(linestrings)
    drhs = cuspatial.GeoSeries(polygons)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_linestring_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))


@pytest.mark.parametrize(
    "linestrings",
    [
        [LineString([(0, 0), (1, 1)]), LineString([(10, 10), (11, 11)])],
        [
            MultiLineString([[(1, 1), (2, 2)], [(3, 3), (4, 4)]]),
            MultiLineString([[(10, 10), (11, 11)], [(12, 12), (13, 13)]]),
        ],
    ],
)
@pytest.mark.parametrize(
    "polygons",
    [
        [
            Polygon([(0, 1), (1, 0), (-1, 0), (0, 1)]),
            Polygon([(-4, -4), (-4, -5), (-5, -5), (-5, -4), (-5, -5)]),
        ],
        [
            MultiPolygon(
                [
                    Polygon([(0, 1), (1, 0), (-1, 0), (0, 1)]),
                    Polygon([(0, 1), (1, 0), (0, -1), (-1, 0), (0, 1)]),
                ]
            ),
            MultiPolygon(
                [
                    Polygon(
                        [(-4, -4), (-4, -5), (-5, -5), (-5, -4), (-5, -5)]
                    ),
                    Polygon([(-2, 0), (-2, -2), (0, -2), (0, 0), (-2, 0)]),
                ]
            ),
        ],
    ],
)
def test_two_pair(linestrings, polygons):
    lhs = gpd.GeoSeries(linestrings)
    rhs = gpd.GeoSeries(polygons)

    dlhs = cuspatial.GeoSeries(linestrings)
    drhs = cuspatial.GeoSeries(polygons)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_linestring_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))


def test_linestring_polygon_large(linestring_generator, polygon_generator):
    N = 100
    linestrings = gpd.GeoSeries(linestring_generator(N, 5))
    polygons = gpd.GeoSeries(polygon_generator(N, 10.0, 3.0))

    dlinestrings = cuspatial.from_geopandas(linestrings)
    dpolygons = cuspatial.from_geopandas(polygons)

    expect = linestrings.distance(polygons)
    got = cuspatial.pairwise_linestring_polygon_distance(
        dlinestrings, dpolygons
    )

    assert_series_equal(got, cudf.Series(expect))
