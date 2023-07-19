# Copyright (c) 2020-2023, NVIDIA CORPORATION

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPoint, Point, Polygon

import cuspatial


def test_point_geom_equals_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.geom_equals(point2)
    expected = gpdpoint1.geom_equals(gpdpoint2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.parametrize(
    "lhs",
    [
        [Point(0, 0), Point(0, 0), Point(0, 0)],
        [Point(1, 1), Point(1, 1), Point(1, 1)],
        [Point(2, 2), Point(2, 2), Point(2, 2)],
    ],
)
def test_3_points_equals_3_points_one_equal(lhs):
    gpdpoint1 = gpd.GeoSeries(lhs)
    gpdpoint2 = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.geom_equals(point2)
    expected = gpdpoint1.geom_equals(gpdpoint2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_points_geom_equals_100_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(100)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(100)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.geom_equals(points2)
    expected = gpdpoints1.geom_equals(gpdpoints2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_geom_equals_linestring():
    gpdline1 = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdline2 = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    line1 = cuspatial.from_geopandas(gpdline1)
    line2 = cuspatial.from_geopandas(gpdline2)
    got = line1.geom_equals(line2)
    expected = gpdline1.geom_equals(gpdline2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_geom_equals_linestring_reversed():
    gpdline1 = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdline2 = gpd.GeoSeries([LineString([(1, 1), (0, 0)])])
    line1 = cuspatial.from_geopandas(gpdline1)
    line2 = cuspatial.from_geopandas(gpdline2)
    got = line1.geom_equals(line2)
    expected = gpdline1.geom_equals(gpdline2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.parametrize(
    "lhs",
    [
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (1, 1)]),
        ],
        [
            LineString([(1, 1), (2, 2)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(1, 1), (2, 2)]),
        ],
        [
            LineString([(2, 2), (3, 3)]),
            LineString([(2, 2), (3, 3)]),
            LineString([(2, 2), (3, 3)]),
        ],
    ],
)
def test_3_linestrings_equals_3_linestrings_one_equal(lhs):
    gpdline1 = gpd.GeoSeries(lhs)
    gpdline2 = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(2, 2), (3, 3)]),
        ]
    )
    line1 = cuspatial.from_geopandas(gpdline1)
    line2 = cuspatial.from_geopandas(gpdline2)
    got = line1.geom_equals(line2)
    expected = gpdline1.geom_equals(gpdline2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_10_linestrings_geom_equals_10_linestrings(linestring_generator):
    gpdlines1 = gpd.GeoSeries([*linestring_generator(10, 5)])
    gpdlines2 = gpd.GeoSeries([*linestring_generator(10, 5)])
    lines1 = cuspatial.from_geopandas(gpdlines1)
    lines2 = cuspatial.from_geopandas(gpdlines2)
    got = lines1.geom_equals(lines2)
    expected = gpdlines1.geom_equals(gpdlines2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_linestrings_geom_equals_100_linestrings(linestring_generator):
    gpdlines1 = gpd.GeoSeries([*linestring_generator(100, 5)])
    gpdlines2 = gpd.GeoSeries([*linestring_generator(100, 5)])
    lines1 = cuspatial.from_geopandas(gpdlines1)
    lines2 = cuspatial.from_geopandas(gpdlines2)
    got = lines1.geom_equals(lines2)
    expected = gpdlines1.geom_equals(gpdlines2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_geom_equals_polygon():
    gpdline = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    line = cuspatial.from_geopandas(gpdline)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = line.geom_equals(polygon)
    expected = gpdline.geom_equals(gpdpolygon)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_linestrings_geom_equals_100_polygons(
    polygon_generator, linestring_generator
):
    gpdlines = gpd.GeoSeries([*linestring_generator(100, 5)])
    gpdpolygons = gpd.GeoSeries([*polygon_generator(100, 0)])
    lines = cuspatial.from_geopandas(gpdlines)
    polygons = cuspatial.from_geopandas(gpdpolygons)
    got = lines.geom_equals(polygons)
    expected = gpdlines.geom_equals(gpdpolygons)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_polygon_geom_equals_linestring():
    gpdline = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    line = cuspatial.from_geopandas(gpdline)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = polygon.geom_equals(line)
    expected = gpdpolygon.geom_equals(gpdline)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_polygons_geom_equals_100_linestrings(
    polygon_generator, linestring_generator
):
    gpdpolygons = gpd.GeoSeries([*polygon_generator(100, 0)])
    gpdlines = gpd.GeoSeries([*linestring_generator(100, 5)])
    polygons = cuspatial.from_geopandas(gpdpolygons)
    lines = cuspatial.from_geopandas(gpdlines)
    got = polygons.geom_equals(lines)
    expected = gpdpolygons.geom_equals(gpdlines)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_point_contains_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.contains_properly(point2)
    expected = gpdpoint1.contains(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_point_not_contains_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(1, 1)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.contains_properly(point2)
    expected = gpdpoint1.contains(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_100_points_contains_100_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(100)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(100)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.contains_properly(points2)
    expected = gpdpoints1.contains(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_point_covers_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.covers(point2)
    expected = gpdpoint1.covers(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_point_not_covers_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(1, 1)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.covers(point2)
    expected = gpdpoint1.covers(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_100_points_covers_100_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(100)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(100)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.covers(points2)
    expected = gpdpoints1.covers(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_point_intersects_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.intersects(point2)
    expected = gpdpoint1.intersects(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_point_not_intersects_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(1, 1)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.intersects(point2)
    expected = gpdpoint1.intersects(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_100_points_intersects_100_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(100)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(100)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.intersects(points2)
    expected = gpdpoints1.intersects(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_point_within_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.within(point2)
    expected = gpdpoint1.within(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_point_not_within_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(1, 1)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.within(point2)
    expected = gpdpoint1.within(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_100_points_within_100_points(point_generator):
    gpdpoints1 = gpd.GeoSeries(point_generator(100))
    gpdpoints2 = gpd.GeoSeries(point_generator(100))
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.within(points2).values_host
    expected = gpdpoints1.within(gpdpoints2).values
    assert (expected == got).all()


def test_point_crosses_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.crosses(point2)
    expected = gpdpoint1.crosses(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_point_not_crosses_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(1, 1)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.crosses(point2)
    expected = gpdpoint1.crosses(gpdpoint2)
    assert (got.values_host == expected.values).all()


@pytest.mark.parametrize(
    "points",
    [
        [Point(0, 0), Point(3, 3), Point(3, 3)],
        [Point(3, 3), Point(1, 1), Point(3, 3)],
        [Point(3, 3), Point(3, 3), Point(2, 2)],
    ],
)
def test_three_points_crosses_three_points(points):
    gpdpoints1 = gpd.GeoSeries(points)
    gpdpoints2 = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.crosses(points2)
    expected = gpdpoints1.crosses(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_100_points_crosses_100_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(100)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(100)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.crosses(points2)
    expected = gpdpoints1.crosses(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_point_overlaps_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.overlaps(point2)
    expected = gpdpoint1.overlaps(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_point_not_overlaps_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(1, 1)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.overlaps(point2)
    expected = gpdpoint1.overlaps(gpdpoint2)
    assert (got.values_host == expected.values).all()


@pytest.mark.parametrize(
    "points",
    [
        [Point(0, 0), Point(3, 3), Point(3, 3)],
        [Point(3, 3), Point(1, 1), Point(3, 3)],
        [Point(3, 3), Point(3, 3), Point(2, 2)],
    ],
)
def test_three_points_overlaps_three_points(points):
    gpdpoints1 = gpd.GeoSeries(points)
    gpdpoints2 = gpd.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.overlaps(points2)
    expected = gpdpoints1.overlaps(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_100_points_overlaps_100_points(point_generator):
    gpdpoint1 = gpd.GeoSeries([*point_generator(100)])
    gpdpoint2 = gpd.GeoSeries([*point_generator(100)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.overlaps(point2)
    expected = gpdpoint1.overlaps(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_multipoint_geom_equals_multipoint():
    gpdpoint1 = gpd.GeoSeries([MultiPoint([(0, 0), (1, 1)])])
    gpdpoint2 = gpd.GeoSeries([MultiPoint([(0, 0), (1, 1)])])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.geom_equals(point2)
    expected = gpdpoint1.geom_equals(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_multipoint_not_geom_equals_multipoint():
    gpdpoint1 = gpd.GeoSeries([MultiPoint([(0, 0), (1, 1)])])
    gpdpoint2 = gpd.GeoSeries([MultiPoint([(0, 1), (1, 1)])])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.geom_equals(point2)
    expected = gpdpoint1.geom_equals(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_100_multipoints_geom_equals_100_multipoints(multipoint_generator):
    gpdpoints1 = gpd.GeoSeries([*multipoint_generator(100, 10)])
    gpdpoints2 = gpd.GeoSeries([*multipoint_generator(100, 10)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.geom_equals(points2)
    expected = gpdpoints1.geom_equals(gpdpoints2)
    assert (got.values_host == expected.values).all()


@pytest.mark.parametrize(
    "lhs",
    [
        [
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
        ],
        [
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 1), (1, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
        ],
        [
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 2), (1, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
        ],
    ],
)
def test_3_multipoints_geom_equals_3_multipoints_one_equal(lhs):
    gpdpoints1 = gpd.GeoSeries(lhs)
    gpdpoints2 = gpd.GeoSeries(
        [
            MultiPoint([(0, 0), (0, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 0), (2, 1)]),
        ]
    )
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.geom_equals(points2)
    expected = gpdpoints1.geom_equals(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_3_multipoints_geom_equals_3_multipoints_misordered():
    gpdpoints1 = gpd.GeoSeries(
        [
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(0, 0), (1, 1)]),
        ]
    )
    gpdpoints2 = gpd.GeoSeries(
        [
            MultiPoint([(1, 1), (0, 0)]),
            MultiPoint([(1, 1), (0, 0)]),
            MultiPoint([(1, 1), (0, 0)]),
        ]
    )
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.geom_equals(points2)
    expected = gpdpoints1.geom_equals(gpdpoints2)
    assert (got.values_host == expected.values).all()


def test_3_linestrings_geom_equals_3_linestrings_misordered():
    gpdline1 = gpd.GeoSeries(
        [
            LineString([(1, 1), (0, 0)]),
            LineString([(2, 2), (1, 1)]),
            LineString([(3, 3), (2, 2)]),
        ]
    )
    gpdline2 = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
            LineString([(2, 2), (3, 3)]),
        ]
    )
    line1 = cuspatial.from_geopandas(gpdline1)
    line2 = cuspatial.from_geopandas(gpdline2)
    got = line1.geom_equals(line2)
    expected = gpdline1.geom_equals(gpdline2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_3_linestrings_geom_equals_3_linestrings_longer():
    gpdline1 = gpd.GeoSeries(
        [
            LineString([(1, 1), (0, 0), (0, 4)]),
            LineString([(2, 2), (1, 1), (0, 4)]),
            LineString([(3, 3), (2, 2), (0, 4)]),
        ]
    )
    gpdline2 = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (0, 4)]),
            LineString([(1, 1), (2, 2), (0, 4)]),
            LineString([(2, 2), (3, 3), (0, 4)]),
        ]
    )
    line1 = cuspatial.from_geopandas(gpdline1)
    line2 = cuspatial.from_geopandas(gpdline2)
    got = line1.geom_equals(line2)
    expected = gpdline1.geom_equals(gpdline2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_pair_linestrings_different_last_two():
    gpdlinestring1 = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (2, 1)]),
        ]
    )
    gpdlinestring2 = gpd.GeoSeries(
        [
            LineString([(0, 0), (2, 1), (1, 1)]),
        ]
    )
    linestring1 = cuspatial.from_geopandas(gpdlinestring1)
    linestring2 = cuspatial.from_geopandas(gpdlinestring2)
    got = linestring1.geom_equals(linestring2)
    expected = gpdlinestring1.geom_equals(gpdlinestring2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.xfail(
    reason="""The current implementation of .contains
conceals this special case. Unsure of the solution."""
)
def test_pair_polygons_different_ordering():
    gpdpoly1 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0.5, 0.5), (0, 1), (0, 0)]),
        ]
    )
    gpdpoly2 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (0.5, 0.5), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ]
    )
    poly1 = cuspatial.from_geopandas(gpdpoly1)
    poly2 = cuspatial.from_geopandas(gpdpoly2)
    got = poly1.geom_equals(poly2)
    expected = gpdpoly1.geom_equals(gpdpoly2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_pair_polygons_different_winding():
    gpdpoly1 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0.5, 0.5), (0, 1), (0, 0)]),
        ]
    )
    gpdpoly2 = gpd.GeoSeries(
        [
            Polygon([(1, 0), (1, 1), (0.5, 0.5), (0, 1), (0, 0), (1, 0)]),
        ]
    )
    poly1 = cuspatial.from_geopandas(gpdpoly1)
    poly2 = cuspatial.from_geopandas(gpdpoly2)
    got = poly1.geom_equals(poly2)
    expected = gpdpoly1.geom_equals(gpdpoly2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_3_polygons_geom_equals_3_polygons_misordered_corrected_vertex():
    gpdpoly1 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
        ]
    )
    gpdpoly2 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),  # Oppositely wound
            Polygon([(1, 1), (0, 1), (0, 0), (1, 1)]),  # Wound by +1 offset
            Polygon([(0, 1), (0, 0), (1, 1), (0, 1)]),  # Wound by -1 offset
        ]
    )
    poly1 = cuspatial.from_geopandas(gpdpoly1)
    poly2 = cuspatial.from_geopandas(gpdpoly2)
    got = poly1.geom_equals(poly2)
    expected = gpdpoly1.geom_equals(gpdpoly2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_polygon_geom_equals_polygon():
    gpdpolygon1 = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    gpdpolygon2 = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    polygon1 = cuspatial.from_geopandas(gpdpolygon1)
    polygon2 = cuspatial.from_geopandas(gpdpolygon2)
    got = polygon1.geom_equals(polygon2)
    expected = gpdpolygon1.geom_equals(gpdpolygon2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_polygon_geom_equals_polygon_swap_inner():
    gpdpolygon1 = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    gpdpolygon2 = gpd.GeoSeries(Polygon([[0, 0], [1, 1], [1, 0], [0, 0]]))
    polygon1 = cuspatial.from_geopandas(gpdpolygon1)
    polygon2 = cuspatial.from_geopandas(gpdpolygon2)
    got = polygon1.geom_equals(polygon2)
    expected = gpdpolygon1.geom_equals(gpdpolygon2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


@pytest.mark.parametrize(
    "lhs",
    [
        [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
        ],
        [
            Polygon([[0, 0], [2, 0], [2, 2], [0, 0]]),
            Polygon([[0, 0], [2, 0], [2, 2], [0, 0]]),
            Polygon([[0, 0], [2, 0], [2, 2], [0, 0]]),
        ],
        [
            Polygon([[0, 0], [3, 0], [3, 3], [0, 0]]),
            Polygon([[0, 0], [3, 0], [3, 3], [0, 0]]),
            Polygon([[0, 0], [3, 0], [3, 3], [0, 0]]),
        ],
    ],
)
def test_3_polygons_geom_equals_3_polygons_one_equal(lhs):
    gpdpolygons1 = gpd.GeoSeries(lhs)
    gpdpolygons2 = gpd.GeoSeries(
        [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[0, 0], [2, 0], [2, 2], [0, 0]]),
            Polygon([[0, 0], [3, 0], [3, 3], [0, 0]]),
        ]
    )
    polygons1 = cuspatial.from_geopandas(gpdpolygons1)
    polygons2 = cuspatial.from_geopandas(gpdpolygons2)
    got = polygons1.geom_equals(polygons2)
    expected = gpdpolygons1.geom_equals(gpdpolygons2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_100_polygons_geom_equals_100_polygons(polygon_generator):
    gpdpolygons1 = gpd.GeoSeries([*polygon_generator(100, 0)])
    gpdpolygons2 = gpd.GeoSeries([*polygon_generator(100, 0)])
    polygons1 = cuspatial.from_geopandas(gpdpolygons1)
    polygons2 = cuspatial.from_geopandas(gpdpolygons2)
    got = polygons1.geom_equals(polygons2)
    expected = gpdpolygons1.geom_equals(gpdpolygons2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_3_polygons_geom_equals_3_polygons_different_sizes():
    gpdpoly1 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),  # Length 5
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        ]
    )
    gpdpoly2 = gpd.GeoSeries(
        [
            Polygon(
                [(0, 0), (1, 1), (1, 0), (0, 0)]
            ),  # Oppositely wound, length 4
            Polygon([(1, 1), (1, 0), (0, 0), (1, 1)]),  # Wound by +1 offset
            Polygon([(1, 0), (0, 0), (1, 1), (1, 0)]),  # Wound by -1 offset
        ]
    )
    poly1 = cuspatial.from_geopandas(gpdpoly1)
    poly2 = cuspatial.from_geopandas(gpdpoly2)
    got = poly1.geom_equals(poly2)
    expected = gpdpoly1.geom_equals(gpdpoly2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_3_polygons_geom_equals_3_polygons_misordered():
    gpdpoly1 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (0, 0)]),
        ]
    )
    gpdpoly2 = gpd.GeoSeries(
        [
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),  # Oppositely wound
            Polygon([(1, 1), (1, 0), (0, 0), (1, 1)]),  # Wound by +1 offset
            Polygon([(1, 0), (0, 0), (1, 1), (1, 0)]),  # Wound by -1 offset
        ]
    )
    poly1 = cuspatial.from_geopandas(gpdpoly1)
    poly2 = cuspatial.from_geopandas(gpdpoly2)
    got = poly1.geom_equals(poly2)
    expected = gpdpoly1.geom_equals(gpdpoly2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_orders():
    gpdlinestring1 = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    gpdlinestring2 = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ]
    )
    linestring1 = cuspatial.from_geopandas(gpdlinestring1)
    linestring2 = cuspatial.from_geopandas(gpdlinestring2)
    got = linestring1.geom_equals(linestring2)
    expected = gpdlinestring1.geom_equals(gpdlinestring2)
    pd.testing.assert_series_equal(expected, got.to_pandas())


def test_linestring_indexes():
    linestring1 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 0), (1, 1), (0, 0)]),
            LineString([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    linestring2 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 0), (1, 1), (0, 0)]),
            LineString([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    index1 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    index2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    linestring1 = linestring1[index1].reset_index(drop=True)
    linestring2 = linestring2[index2].reset_index(drop=True)

    gpdlinestring1 = linestring1.to_geopandas()
    gpdlinestring2 = linestring2.to_geopandas()
    got = linestring1.geom_equals(linestring2)
    expected = gpdlinestring1.geom_equals(gpdlinestring2)
    pd.testing.assert_series_equal(expected, got.to_pandas())
