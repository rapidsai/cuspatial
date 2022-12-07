import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

import cuspatial


def test_point_equals_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.equals(point2)
    expected = gpdpoint1.equals(gpdpoint2)
    assert got == expected


def test_31_points_equals_31_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(31)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(31)])
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.equals(points2)
    expected = gpdpoints1.equals(gpdpoints2)
    assert got == expected


def test_linestring_equals_linestring():
    gpdline1 = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdline2 = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    line1 = cuspatial.from_geopandas(gpdline1)
    line2 = cuspatial.from_geopandas(gpdline2)
    got = line1.equals(line2)
    expected = gpdline1.equals(gpdline2)
    assert got == expected


def test_31_linestrings_equals_31_linestrings(linestring_generator):
    gpdlines1 = gpd.GeoSeries([*linestring_generator(31, 5)])
    gpdlines2 = gpd.GeoSeries([*linestring_generator(31, 5)])
    lines1 = cuspatial.from_geopandas(gpdlines1)
    lines2 = cuspatial.from_geopandas(gpdlines2)
    got = lines1.equals(lines2)
    expected = gpdlines1.equals(gpdlines2)
    assert got == expected


def test_linestring_equals_polygon():
    gpdline = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    line = cuspatial.from_geopandas(gpdline)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = line.equals(polygon)
    expected = gpdline.equals(gpdpolygon)
    assert got == expected


def test_31_linestrings_equals_31_polygons(
    polygon_generator, linestring_generator
):
    gpdlines = gpd.GeoSeries([*linestring_generator(31, 5)])
    gpdpolygons = gpd.GeoSeries([*polygon_generator(31, 0)])
    lines = cuspatial.from_geopandas(gpdlines)
    polygons = cuspatial.from_geopandas(gpdpolygons)
    got = lines.equals(polygons)
    expected = gpdlines.equals(gpdpolygons)
    assert got == expected


def test_polygon_equals_linestring():
    gpdline = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    line = cuspatial.from_geopandas(gpdline)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = polygon.equals(line)
    expected = gpdpolygon.equals(gpdline)
    assert got == expected


def test_31_polygons_equals_31_linestrings(
    polygon_generator, linestring_generator
):
    gpdpolygons = gpd.GeoSeries([*polygon_generator(31, 0)])
    gpdlines = gpd.GeoSeries([*linestring_generator(31, 5)])
    polygons = cuspatial.from_geopandas(gpdpolygons)
    lines = cuspatial.from_geopandas(gpdlines)
    got = polygons.equals(lines)
    expected = gpdpolygons.equals(gpdlines)
    assert got == expected


def test_polygon_equals_polygon():
    gpdpolygon1 = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    gpdpolygon2 = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    polygon1 = cuspatial.from_geopandas(gpdpolygon1)
    polygon2 = cuspatial.from_geopandas(gpdpolygon2)
    got = polygon1.equals(polygon2)
    expected = gpdpolygon1.equals(gpdpolygon2)
    assert got == expected


def test_31_polygons_equals_31_polygons(polygon_generator):
    gpdpolygons1 = gpd.GeoSeries([*polygon_generator(31, 0)])
    gpdpolygons2 = gpd.GeoSeries([*polygon_generator(31, 0)])
    polygons1 = cuspatial.from_geopandas(gpdpolygons1)
    polygons2 = cuspatial.from_geopandas(gpdpolygons2)
    got = polygons1.equals(polygons2)
    expected = gpdpolygons1.equals(gpdpolygons2)
    assert got == expected


def test_point_contains_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.contains_properly(point2)
    expected = gpdpoint1.contains(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_31_points_contains_31_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(31)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(31)])
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


def test_31_points_covers_31_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(31)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(31)])
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


def test_31_points_intersects_31_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(31)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(31)])
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


def test_31_points_within_31_points(point_generator):
    gpdpoints1 = gpd.GeoSeries(point_generator(31))
    gpdpoints2 = gpd.GeoSeries(point_generator(31))
    points1 = cuspatial.from_geopandas(gpdpoints1)
    points2 = cuspatial.from_geopandas(gpdpoints2)
    got = points1.within(points2).values_host
    expected = gpdpoints1.within(gpdpoints2).values
    assert (got == expected).all()


def test_point_crosses_point():
    gpdpoint1 = gpd.GeoSeries([Point(0, 0)])
    gpdpoint2 = gpd.GeoSeries([Point(0, 0)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.crosses(point2)
    expected = gpdpoint1.crosses(gpdpoint2)
    assert (got.values_host == expected.values).all()


def test_31_points_crosses_31_points(point_generator):
    gpdpoints1 = gpd.GeoSeries([*point_generator(31)])
    gpdpoints2 = gpd.GeoSeries([*point_generator(31)])
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


def test_31_points_overlaps_31_points(point_generator):
    gpdpoint1 = gpd.GeoSeries([*point_generator(31)])
    gpdpoint2 = gpd.GeoSeries([*point_generator(31)])
    point1 = cuspatial.from_geopandas(gpdpoint1)
    point2 = cuspatial.from_geopandas(gpdpoint2)
    got = point1.overlaps(point2)
    expected = gpdpoint1.overlaps(gpdpoint2)
    assert (got.values_host == expected.values).all()
