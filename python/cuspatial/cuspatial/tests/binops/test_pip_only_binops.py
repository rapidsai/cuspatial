import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

import cuspatial

"""
# Needs intersection
def test_polygon_crosses_point():
    gpdpolygon = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    gpdpoint = gpd.GeoSeries([Point(0.5, 0.5)])
    polygon = cuspatial.from_geopandas(gpdpolygon)
    point = cuspatial.from_geopandas(gpdpoint)
    got = polygon.crosses(point).values_host
    expected = gpdpolygon.crosses(gpdpoint).values
    assert (got == expected).all()


# Edge case - point intersects polygon boundary
def test_point_intersects_polygon():
    gpdpoint = gpd.GeoSeries([Point(0, 0)])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]))
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = point.intersects(polygon).values_host
    expected = gpdpoint.intersects(gpdpolygon).values
    assert (got == expected).all()
"""


# Needs any instead of all.
def test_polygon_overlaps_point():
    gpdpolygon = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    gpdpoint = gpd.GeoSeries([Point(0.5, 0.5)])
    polygon = cuspatial.from_geopandas(gpdpolygon)
    point = cuspatial.from_geopandas(gpdpoint)
    got = polygon.overlaps(point).values_host
    expected = gpdpolygon.overlaps(gpdpoint).values
    assert (got == expected).all()


# Needs any instead of all.
def test_polygon_overlaps_polygon():
    gpdpolygon1 = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    gpdpolygon2 = gpd.GeoSeries(
        Polygon([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5], [0.5, 0.5]])
    )
    polygon1 = cuspatial.from_geopandas(gpdpolygon1)
    polygon2 = cuspatial.from_geopandas(gpdpolygon2)
    got = polygon1.overlaps(polygon2).values_host
    expected = gpdpolygon1.overlaps(gpdpolygon2).values
    assert (got == expected).all()


# Interior case - point intersects polygon interior
def test_point_intersects_polygon_interior():
    gpdpoint = gpd.GeoSeries([Point(0.5, 0.5)])
    gpdpolygon = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = point.intersects(polygon).values_host
    expected = gpdpoint.intersects(gpdpolygon).values
    assert (got == expected).all()


def test_point_within_polygon():
    gpdpoint = gpd.GeoSeries([Point(0, 0)])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    point = cuspatial.from_geopandas(gpdpoint)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = point.within(polygon).values_host
    expected = gpdpoint.within(gpdpolygon).values
    assert (got == expected).all()


def test_linestring_within_polygon():
    gpdline = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    gpdpolygon = gpd.GeoSeries(Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]))
    line = cuspatial.from_geopandas(gpdline)
    polygon = cuspatial.from_geopandas(gpdpolygon)
    got = line.within(polygon).values_host
    expected = gpdline.within(gpdpolygon).values
    assert (got == expected).all()


def test_polygon_within_polygon():
    gpdpolygon1 = gpd.GeoSeries(
        Polygon([[0, 0], [-1, 1], [1, 1], [1, -2], [0, 0]])
    )
    gpdpolygon2 = gpd.GeoSeries(Polygon([[-1, -1], [-2, 2], [2, 2], [2, -2]]))
    polygon1 = cuspatial.from_geopandas(gpdpolygon1)
    polygon2 = cuspatial.from_geopandas(gpdpolygon2)
    got = polygon1.within(polygon2).values_host
    expected = gpdpolygon1.within(gpdpolygon2).values
    assert (got == expected).all()


def test_polygon_overlaps_linestring():
    gpdpolygon = gpd.GeoSeries(
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])
    )
    gpdline = gpd.GeoSeries([LineString([(0.5, 0.5), (1.5, 1.5)])])
    polygon = cuspatial.from_geopandas(gpdpolygon)
    line = cuspatial.from_geopandas(gpdline)
    got = polygon.overlaps(line).values_host
    expected = gpdpolygon.overlaps(gpdline).values
    assert (got == expected).all()
