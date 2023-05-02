# Copyright (c) 2023, NVIDIA CORPORATION.

import cupy as cp
import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Polygon

import cudf
from cudf.testing import assert_series_equal

import cuspatial


def test_polygon_empty():
    lhs = cuspatial.GeoSeries.from_polygons_xy([], [0], [0], [0])
    rhs = cuspatial.GeoSeries.from_polygons_xy([], [0], [0], [0])

    got = cuspatial.pairwise_polygon_distance(lhs, rhs)

    expect = cudf.Series([], dtype="f8")

    assert_series_equal(got, expect)


@pytest.mark.parametrize(
    "polygons1",
    [
        [Polygon([(10, 11), (11, 10), (11, 11), (10, 11)])],
        [
            MultiPolygon(
                [
                    Polygon([(12, 10), (11, 10), (11, 11), (12, 10)]),
                    Polygon([(11, 10), (12, 10), (11, 11), (11, 10)]),
                ]
            )
        ],
    ],
)
@pytest.mark.parametrize(
    "polygons2",
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
def test_one_pair(polygons1, polygons2):
    lhs = gpd.GeoSeries(polygons1)
    rhs = gpd.GeoSeries(polygons2)

    dlhs = cuspatial.GeoSeries(polygons1)
    drhs = cuspatial.GeoSeries(polygons2)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))


@pytest.mark.parametrize(
    "polygons1",
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
@pytest.mark.parametrize(
    "polygons2",
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
def test_two_pair(polygons1, polygons2):
    lhs = gpd.GeoSeries(polygons1)
    rhs = gpd.GeoSeries(polygons2)

    dlhs = cuspatial.GeoSeries(polygons1)
    drhs = cuspatial.GeoSeries(polygons2)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))


def test_linestring_polygon_large(polygon_generator):
    N = 100
    polygons1 = gpd.GeoSeries(polygon_generator(N, 20.0, 5.0))
    polygons2 = gpd.GeoSeries(polygon_generator(N, 10.0, 3.0))

    dpolygons1 = cuspatial.from_geopandas(polygons1)
    dpolygons2 = cuspatial.from_geopandas(polygons2)

    expect = polygons1.distance(polygons2)
    got = cuspatial.pairwise_polygon_distance(dpolygons1, dpolygons2)

    assert_series_equal(got, cudf.Series(expect))


def test_point_polygon_geoboundaries(naturalearth_lowres):
    N = 50

    lhs = naturalearth_lowres.geometry[:N].reset_index(drop=True)
    rhs = naturalearth_lowres.geometry[N : 2 * N].reset_index(drop=True)
    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_polygon_distance(
        cuspatial.GeoSeries(lhs), cuspatial.GeoSeries(rhs)
    )
    assert_series_equal(cudf.Series(expect), got)


def test_self_distance(polygon_generator):
    N = 100
    polygons = gpd.GeoSeries(polygon_generator(N, 20.0, 5.0))
    polygons = cuspatial.from_geopandas(polygons)
    got = cuspatial.pairwise_polygon_distance(polygons, polygons)
    expect = cudf.Series(cp.zeros((N,)))

    assert_series_equal(got, expect)


def test_touching_distance():
    polygons1 = [Polygon([(0, 0), (1, 1), (1, 0), (0, 0)])]
    polygons2 = [Polygon([(1, 0.5), (2, 0), (3, 0.5), (1, 0.5)])]

    got = cuspatial.pairwise_polygon_distance(
        cuspatial.GeoSeries(polygons1), cuspatial.GeoSeries(polygons2)
    )

    expect = gpd.GeoSeries(polygons1).distance(gpd.GeoSeries(polygons2))

    assert_series_equal(got, cudf.Series(expect))


def test_distance_one():
    polygons1 = [Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])]

    polygons2 = [Polygon([(0, 0), (0, 1), (-1, 1), (-1, 0), (0, 0)])]

    got = cuspatial.pairwise_polygon_distance(
        cuspatial.GeoSeries(polygons1), cuspatial.GeoSeries(polygons2)
    )

    expect = gpd.GeoSeries(polygons1).distance(gpd.GeoSeries(polygons2))

    assert_series_equal(got, cudf.Series(expect))
