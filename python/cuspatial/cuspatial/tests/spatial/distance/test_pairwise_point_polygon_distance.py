# Copyright (c) 2023, NVIDIA CORPORATION.

import geopandas as gpd
import pytest
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

import cudf
from cudf.testing import assert_series_equal

import cuspatial


def test_point_polygon_empty():
    lhs = cuspatial.GeoSeries.from_points_xy([])
    rhs = cuspatial.GeoSeries.from_polygons_xy([], [0], [0], [0])

    got = cuspatial.pairwise_point_polygon_distance(lhs, rhs)

    expect = cudf.Series([], dtype="f8")

    assert_series_equal(got, expect)


def test_multipoint_polygon_empty():
    lhs = cuspatial.GeoSeries.from_multipoints_xy([], [0])
    rhs = cuspatial.GeoSeries.from_polygons_xy([], [0], [0], [0])

    got = cuspatial.pairwise_point_polygon_distance(lhs, rhs)

    expect = cudf.Series([], dtype="f8")

    assert_series_equal(got, expect)


@pytest.mark.parametrize(
    "points", [[Point(0, 0)], [MultiPoint([(1, 1), (2, 2)])]]
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
def test_one_pair(points, polygons):
    lhs = gpd.GeoSeries(points)
    rhs = gpd.GeoSeries(polygons)

    dlhs = cuspatial.GeoSeries(points)
    drhs = cuspatial.GeoSeries(polygons)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_point_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))


@pytest.mark.parametrize(
    "points",
    [
        [Point(0, 0), Point(3, -3)],
        [MultiPoint([(1, 1), (2, 2)]), MultiPoint([(3, 3), (4, 4)])],
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
def test_two_pair(points, polygons):
    lhs = gpd.GeoSeries(points)
    rhs = gpd.GeoSeries(polygons)

    dlhs = cuspatial.GeoSeries(points)
    drhs = cuspatial.GeoSeries(polygons)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_point_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))


def test_point_polygon_large(point_generator, polygon_generator):
    N = 100
    points = gpd.GeoSeries(point_generator(N))
    polygons = gpd.GeoSeries(polygon_generator(N, 1.0, 1.5))

    dpoints = cuspatial.from_geopandas(points)
    dpolygons = cuspatial.from_geopandas(polygons)

    expect = points.distance(polygons)
    got = cuspatial.pairwise_point_polygon_distance(dpoints, dpolygons)

    assert_series_equal(got, cudf.Series(expect))


def test_point_polygon_geocities(naturalearth_cities, naturalearth_lowres):
    N = 100
    gpu_cities = cuspatial.from_geopandas(naturalearth_cities.geometry)
    gpu_countries = cuspatial.from_geopandas(naturalearth_lowres.geometry)

    expect = naturalearth_cities.geometry[:N].distance(
        naturalearth_lowres.geometry[:N]
    )

    got = cuspatial.pairwise_point_polygon_distance(
        gpu_cities[:N], gpu_countries[:N]
    )

    assert_series_equal(cudf.Series(expect), got)
