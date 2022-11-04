import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

import cuspatial

"""
[
    Point([0.6, 0.06]),
    Polygon([[0, 0], [10, 1], [1, 1], [0, 0]]),
    False,
],
[
    Point([3.333, 1.111]),
    Polygon([[6, 2], [3, 1], [3, 4], [6, 2]]),
    True,
],
"""


@pytest.mark.parametrize(
    "point, polygon, expects",
    [
        # unique failure cases identified by @mharris
        [
            Point([0.66, 0.006]),
            Polygon([[0, 0], [10, 1], [1, 1], [0, 0]]),
            False,
        ],
        [
            Point([0.666, 0.0006]),
            Polygon([[0, 0], [10, 1], [1, 1], [0, 0]]),
            False,
        ],
        [Point([3.3, 1.1]), Polygon([[6, 2], [3, 1], [3, 4], [6, 2]]), True],
        [Point([3.33, 1.11]), Polygon([[6, 2], [3, 1], [3, 4], [6, 2]]), True],
    ],
)
def test_float_precision_limits(point, polygon, expects):
    gpdpoint = gpd.GeoSeries(point)
    gpdpolygon = gpd.GeoSeries(polygon)
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdpoint)
    got = polygon.contains(point).to_pandas()
    pd.testing.assert_series_equal(expected, got)
    assert got.values == expects


clockwiseTriangle = Polygon([[0, 0], [0, 1], [1, 1], [0, 0]])
clockwiseSquare = Polygon(
    [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]]
)


@pytest.mark.parametrize(
    "point, polygon, expects",
    [
        [Point([-0.5, -0.5]), clockwiseSquare, False],
        [Point([-0.5, 0.5]), clockwiseSquare, False],
        [Point([0.5, 0.5]), clockwiseSquare, False],
        [Point([0.5, -0.5]), clockwiseSquare, False],
        # clockwise square, should be true
        [Point([-0.5, 0.0]), clockwiseSquare, False],
        [Point([0.0, 0.5]), clockwiseSquare, False],
        [Point([0.5, 0.0]), clockwiseSquare, False],
        [Point([0.0, -0.5]), clockwiseSquare, False],
        # wound clockwise, should be false
        [Point([0, 0]), clockwiseTriangle, False],
        [Point([0.0, 1.0]), clockwiseTriangle, False],
        [Point([1.0, 1.0]), clockwiseTriangle, False],
        [Point([0.0, 0.5]), clockwiseTriangle, False],
        [Point([0.5, 0.5]), clockwiseTriangle, False],
        [Point([0.5, 1]), clockwiseTriangle, False],
        # wound clockwise, should be true
        [Point([0.25, 0.5]), clockwiseTriangle, True],
        [Point([0.75, 0.9]), clockwiseTriangle, True],
        # wound counter clockwise, should be false
        [Point([0.0, 0.0]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), False],
        [Point([1.0, 0.0]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), False],
        [Point([1.0, 1.0]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), False],
        [Point([0.5, 0.0]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), False],
        [Point([0.5, 0.5]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), False],
        # wound counter clockwise, should be true
        [Point([0.5, 0.25]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), True],
        [Point([0.9, 0.75]), Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]), True],
    ],
)
def test_point_in_polygon(point, polygon, expects):
    gpdpoint = gpd.GeoSeries(point)
    gpdpolygon = gpd.GeoSeries(polygon)
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdpoint)
    got = polygon.contains(point).to_pandas()
    pd.testing.assert_series_equal(expected, got)
    assert got.values == expects


def test_two_points_one_polygon():
    gpdpoint = gpd.GeoSeries([Point(0, 0), Point(0, 0)])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdpoint)
    got = polygon.contains(point).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_one_point_two_polygons():
    gpdpoint = gpd.GeoSeries([Point(0, 0)])
    gpdpolygon = gpd.GeoSeries(
        [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
        ]
    )
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdpoint)
    got = polygon.contains(point).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_ten_pair_points(point_generator, polygon_generator):
    gpdpoints = gpd.GeoSeries([*point_generator(10)])
    gpdpolygons = gpd.GeoSeries([*polygon_generator(10, 0)])
    points = cuspatial.from_geopandas(gpdpoints)
    polygons = cuspatial.from_geopandas(gpdpolygons)
    expected = gpdpolygons.contains(gpdpoints)
    got = polygons.contains(points).to_pandas()
    pd.testing.assert_series_equal(expected, got)


@pytest.mark.xfail(
    reason="We don't support intersection yet to check for crossings."
)
def test_one_polygon_with_hole_one_linestring_crossing_it(
    linestring_generator,
):
    gpdlinestring = gpd.GeoSeries(LineString([[0.5, 2.0], [3.5, 2.0]]))
    gpdpolygon = gpd.GeoSeries(
        Polygon(
            (
                [0, 0],
                [0, 4],
                [4, 4],
                [4, 0],
                [0, 0],
            ),
            [
                (
                    [1, 1],
                    [1, 3],
                    [3, 3],
                    [3, 1],
                    [1, 1],
                )
            ],
        )
    )
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdlinestring)
    got = polygons.contains(linestring).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_one_polygon_with_hole_one_linestring_inside_it(linestring_generator):
    gpdlinestring = gpd.GeoSeries(LineString([[1.5, 2.0], [2.5, 2.0]]))
    gpdpolygon = gpd.GeoSeries(
        Polygon(
            (
                [0, 0],
                [0, 4],
                [4, 4],
                [4, 0],
                [0, 0],
            ),
            [
                (
                    [1, 1],
                    [1, 3],
                    [3, 3],
                    [3, 1],
                    [1, 1],
                )
            ],
        )
    )
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdlinestring)
    got = polygons.contains(linestring).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_one_polygon_one_linestring(linestring_generator):
    gpdlinestring = gpd.GeoSeries([*linestring_generator(1, 4)])
    gpdpolygon = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdlinestring)
    got = polygons.contains(linestring).to_pandas()
    pd.testing.assert_series_equal(expected, got)


@pytest.mark.xfail(reason="The boundaries share points, so pip is impossible.")
def test_one_polygon_one_linestring_crosses_the_diagonal(linestring_generator):
    gpdlinestring = gpd.GeoSeries(LineString([[0, 0], [1, 1]]))
    gpdpolygon = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdlinestring)
    got = polygons.contains(linestring).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_four_polygons_four_linestrings(linestring_generator):
    gpdlinestring = gpd.GeoSeries(
        [
            LineString([[0.35, 0.35], [0.35, 0.65]]),
            LineString([[0.25, 0.25], [0.25, 0.75]]),
            LineString([[0.15, 0.15], [0.15, 0.85]]),
            LineString([[0.05, 0.05], [0.05, 0.95]]),
        ]
    )
    gpdpolygon = gpd.GeoSeries(
        [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
        ]
    )
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygon)
    expected = gpdpolygon.contains(gpdlinestring)
    got = polygons.contains(linestring).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_max_polygons_max_linestrings(linestring_generator, polygon_generator):
    gpdlinestring = gpd.GeoSeries([*linestring_generator(31, 3)])
    gpdpolygons = gpd.GeoSeries([*polygon_generator(31, 0)])
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygons)
    expected = gpdpolygons.contains(gpdlinestring)
    got = polygons.contains(linestring).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_one_polygon_one_polygon(polygon_generator):
    gpdlhs = gpd.GeoSeries(Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]))
    gpdrhs = gpd.GeoSeries([*polygon_generator(1, 0)])
    rhs = cuspatial.from_geopandas(gpdrhs)
    lhs = cuspatial.from_geopandas(gpdlhs)
    expected = gpdlhs.contains(gpdrhs)
    got = lhs.contains(rhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)
    expected = gpdrhs.contains(gpdlhs)
    got = rhs.contains(lhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)


@pytest.mark.xfail(
    reason="The polygons share numerous boundaries, so pip needs intersection."
)
def test_manual_polygons():
    gpdlhs = gpd.GeoSeries([Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))) * 6])
    gpdrhs = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
            Polygon(((-10, -2), (-10, 2), (-6, 2), (-6, -2))),
            Polygon(((-2, 8), (-2, 12), (2, 12), (2, 8))),
            Polygon(((6, 0), (8, 2), (10, 0), (8, -2))),
            Polygon(((-2, -8), (-2, -4), (2, -4), (2, -8))),
        ]
    )
    rhs = cuspatial.from_geopandas(gpdrhs)
    lhs = cuspatial.from_geopandas(gpdlhs)
    expected = gpdlhs.contains(gpdrhs)
    got = lhs.contains(rhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)
    expected = gpdrhs.contains(gpdlhs)
    got = rhs.contains(lhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_max_polygons_max_polygons(simple_polygon_generator):
    gpdlhs = gpd.GeoSeries([*simple_polygon_generator(31, 1, 3)])
    gpdrhs = gpd.GeoSeries([*simple_polygon_generator(31, 1.49, 2)])
    rhs = cuspatial.from_geopandas(gpdrhs)
    lhs = cuspatial.from_geopandas(gpdlhs)
    expected = gpdlhs.contains(gpdrhs)
    got = lhs.contains(rhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)
    expected = gpdrhs.contains(gpdlhs)
    got = rhs.contains(lhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_one_polygon_one_multipoint(multipoint_generator, polygon_generator):
    gpdlhs = gpd.GeoSeries([*polygon_generator(1, 0)])
    gpdrhs = gpd.GeoSeries([*multipoint_generator(1, 5)])
    rhs = cuspatial.from_geopandas(gpdrhs)
    lhs = cuspatial.from_geopandas(gpdlhs)
    expected = gpdlhs.contains(gpdrhs)
    got = lhs.contains(rhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)


def test_max_polygons_max_multipoints(multipoint_generator, polygon_generator):
    gpdlhs = gpd.GeoSeries([*polygon_generator(31, 0, 1)])
    gpdrhs = gpd.GeoSeries([*multipoint_generator(31, 10)])
    rhs = cuspatial.from_geopandas(gpdrhs)
    lhs = cuspatial.from_geopandas(gpdlhs)
    expected = gpdlhs.contains(gpdrhs)
    got = lhs.contains(rhs).to_pandas()
    pd.testing.assert_series_equal(expected, got)
