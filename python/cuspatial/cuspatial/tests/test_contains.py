import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

import cuspatial

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
    result = polygon.contains(point)
    assert gpdpolygon.contains(gpdpoint).values == result.values_host
    assert result.values_host[0] == expects


def test_two_points_one_polygon():
    gpdpoint = gpd.GeoSeries([Point(0, 0), Point(0, 0)])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    assert (
        gpdpolygon.contains(gpdpoint).values
        == polygon.contains(point).values_host
    ).all()


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
    assert (
        gpdpolygon.contains(gpdpoint).values
        == polygon.contains(point).values_host
    ).all()


def test_ten_pair_points(point_generator, polygon_generator):
    gpdpoints = gpd.GeoSeries([*point_generator(10)])
    gpdpolygons = gpd.GeoSeries([*polygon_generator(10, 0)])
    points = cuspatial.from_geopandas(gpdpoints)
    polygons = cuspatial.from_geopandas(gpdpolygons)
    assert (
        gpdpolygons.contains(gpdpoints).values
        == polygons.contains(points).values_host
    ).all()


def test_one_polygon_one_linestring(linestring_generator):
    gpdlinestring = gpd.GeoSeries([*linestring_generator(1, 3)])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]))
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygon)
    assert (
        gpdpolygon.contains(gpdlinestring).values
        == polygons.contains(linestring).values_host
    ).all()


"""
def test_onehundred_polygons_one_linestring(
    linestring_generator,
    polygon_generator
):
    gpdlinestring = gpd.GeoSeries([*linestring_generator(1, 3)])
    gpdpolygons = gpd.GeoSeries([*polygon_generator(100, 0)])
    linestring = cuspatial.from_geopandas(gpdlinestring)
    polygons = cuspatial.from_geopandas(gpdpolygons)
    assert (
        gpdpolygons.contains(gpdlinestring).values
        == polygons.contains(linestring).values_host
    ).all()
"""
