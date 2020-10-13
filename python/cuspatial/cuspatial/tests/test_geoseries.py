# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)

from cudf.tests.utils import assert_eq

import cuspatial
from cuspatial.geometry.geoseries import (
    cuPoint,
    cuLineString,
)


@pytest.fixture
def gs():
    g0 = Point(1, 2)
    g1 = MultiPoint(((3, 4), (5, 6)))
    g2 = MultiPoint(((5, 6), (7, 8)))
    g3 = Point(9, 10)
    g4 = LineString(((11, 12), (13, 14)))
    g5 = MultiLineString((((15, 16), (17, 18)), ((19, 20), (21, 22))))
    g6 = MultiLineString((((23, 24), (25, 26)), ((27, 28), (29, 30))))
    g7 = LineString(((31, 32), (33, 34)))
    g8 = Polygon(((35, 36), (37, 38), (39, 40), (41, 42)),)
    g9 = MultiPolygon(
        [
            (
                ((43, 44), (45, 46), (47, 48)),
                [((49, 50), (51, 52), (53, 54))],
            ),
            (
                ((55, 56), (57, 58), (59, 60)),
                [((61, 62), (63, 64), (65, 66))],
            ),
        ]
    )
    g10 = MultiPolygon(
        [
            (
                ((67, 68), (69, 70), (71, 72)),
                [((73, 74), (75, 76), (77, 78))],
            ),
            (
                ((79, 80), (81, 82), (83, 84)),
                [
                    ((85, 86), (87, 88), (89, 90)),
                    ((91, 92), (93, 94), (95, 96)),
                ],
            ),
        ]
    )
    g11 = Polygon(
        ((97, 98), (99, 101), (102, 103), (104, 105)),
        [((106, 107), (108, 109), (110, 111), (112, 113))],
    )
    gs = gpd.GeoSeries([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
    return gs


@pytest.fixture
def gs_sorted(gs):
    result = pd.concat(
        [
            gs[gs.type == "Point"],
            gs[gs.type == "MultiPoint"],
            gs[gs.type == "LineString"],
            gs[gs.type == "MultiLineString"],
            gs[gs.type == "Polygon"],
            gs[gs.type == "MultiPolygon"],
        ]
    )
    return result.reset_index(drop=True)


def to_shapely(obj):
    if isinstance(obj, gpd.GeoSeries) and len(obj) == 1:
        return obj[0]
    if isinstance(obj, (cuPoint, cuLineString)):
        return obj.to_shapely()
    return obj


def assert_eq_point(p1, p2):
    p1 = to_shapely(p1)
    p2 = to_shapely(p2)
    assert type(p1) == type(p2)
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.has_z == p2.has_z
    if p1.has_z:
        assert p1.z == p2.z
    assert True


def assert_eq_multipoint(p1, p2):
    p1 = to_shapely(p1)
    p2 = to_shapely(p2)
    assert type(p1) == type(p2)
    assert len(p1) == len(p2)
    for i in range(len(p1)):
        assert_eq_point(p1[i], p2[i])


def assert_eq_linestring(p1, p2):
    p1 = to_shapely(p1)
    p2 = to_shapely(p2)
    assert type(p1) == type(p2)
    assert len(p1.coords) == len(p2.coords)
    for i in range(len(p1.coords)):
        assert_eq(p1.coords[i], p2.coords[i])


def assert_eq_multilinestring(p1, p2):
    p1 = to_shapely(p1)
    p2 = to_shapely(p2)
    for i in range(len(p1)):
        assert_eq_linestring(p1[i], p2[i])


def assert_eq_polygon(p1, p2):
    p1 = to_shapely(p1)
    p2 = to_shapely(p2)
    if not p1.equals(p2):
        raise ValueError


def assert_eq_multipolygon(p1, p2):
    p1 = to_shapely(p1)
    p2 = to_shapely(p2)
    if not p1.equals(p2):
        raise ValueError


def assert_eq_geo(geo1, geo2):
    geo1 = to_shapely(geo1)
    geo2 = to_shapely(geo2)
    if type(geo1) != type(geo2):
        assert TypeError
    if isinstance(geo1, Point):
        assert_eq_point(geo1, geo2)
    elif isinstance(geo1, MultiPoint):
        assert_eq_multipoint(geo1, geo2)
    elif isinstance(geo1, LineString):
        assert_eq_linestring(geo1, geo2)
    elif isinstance(geo1, MultiLineString):
        assert_eq_multilinestring(geo1, geo2)
    elif isinstance(geo1, Polygon):
        assert_eq_polygon(geo1, geo2)
    elif isinstance(geo1, MultiPolygon):
        assert_eq_multipolygon(geo1, geo2)
    else:
        raise TypeError


def test_getitem_points():
    p0 = Point([1, 2])
    p1 = Point([3, 4])
    p2 = Point([5, 6])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_point(cus[0], p0)
    assert_eq_point(cus[1], p1)
    assert_eq_point(cus[2], p2)


def test_getitem_lines():
    p0 = LineString([[1, 2], [3, 4]])
    p1 = LineString([[1, 2], [3, 4], [5, 6], [7, 8]])
    p2 = LineString([[1, 2], [3, 4], [5, 6]])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0], p0)
    assert_eq_linestring(cus[1], p1)
    assert_eq_linestring(cus[2], p2)


@pytest.mark.parametrize("series_slice", list(np.arange(10)) +
        [slice(0, 10, 1)] +
        [slice(0, 3, 1)] +
        [slice(3, 6, 1)] + 
        [slice(6, 9, 1)]
)
def test_size(gs, series_slice):
    geometries = gs[series_slice]
    gi = gpd.GeoSeries(geometries)
    cugs = cuspatial.from_geopandas(gi)
    assert(len(gi) == len(cugs))
    

@pytest.mark.parametrize("series_slice", list(np.arange(10)) +
        [slice(0, 10, 1)] +
        [slice(0, 3, 1)] +
        [slice(3, 6, 1)] + 
        [slice(6, 9, 1)]
)
def test_to_shapely(gs, series_slice):
    geometries = gs[series_slice]
    gi = gpd.GeoSeries(geometries)
    cugs = cuspatial.from_geopandas(gi)
    assert_eq_geo(gi, cugs.to_geopandas())
