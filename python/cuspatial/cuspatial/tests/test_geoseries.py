# Copyright (c) 2020-2021, NVIDIA CORPORATION.

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
from shapely.affinity import rotate

from cudf.tests.utils import assert_eq

import cuspatial

np.random.seed(0)


def random_polygon(distance_from_origin):
    outer = Point(distance_from_origin * 2, 0).buffer(1)
    inners = []
    for i in range(np.random.randint(1, 4)):
        inner = Point(distance_from_origin + i * 0.1, 0).buffer(0.01)
        inners.append(inner)
    together = Polygon(outer, inners)
    result = rotate(together, np.random.random() * 2 * np.pi, use_radians=True)
    return result


def random_multipolygon(size):
    polygons = []
    for i in range(size):
        polygons.append(random_polygon(i))
    result = MultiPolygon(polygons)
    return result


def generator(size, has_z=False):
    obj_type = np.random.randint(1, 7)
    if obj_type == 1:
        return Point(np.random.random(2))
    else:
        if obj_type == 2:
            points = np.random.random(size * 2).reshape(size, 2)
            return MultiPoint(points)
        elif obj_type == 3:
            points = np.random.random(size * 2).reshape(size, 2)
            return LineString(points)
        elif obj_type == 4:
            num_lines = np.random.randint(3, np.ceil(np.sqrt(size)) + 3)
            points = np.random.random(num_lines * size * 2).reshape(
                num_lines, size, 2
            )
            return MultiLineString(tuple(points))
        elif obj_type == 5:
            return random_polygon(size)
        elif obj_type == 6:
            return random_multipolygon(size)


def assert_eq_point(p1, p2):
    assert type(p1) == type(p2)
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.has_z == p2.has_z
    if p1.has_z:
        assert p1.z == p2.z
    assert True


def assert_eq_multipoint(p1, p2):
    assert type(p1) == type(p2)
    assert len(p1) == len(p2)
    for i in range(len(p1)):
        assert_eq_point(p1[i], p2[i])


def assert_eq_linestring(p1, p2):
    assert type(p1) == type(p2)
    assert len(p1.coords) == len(p2.coords)
    for i in range(len(p1.coords)):
        assert_eq(p1.coords[i], p2.coords[i])


def assert_eq_multilinestring(p1, p2):
    for i in range(len(p1)):
        assert_eq_linestring(p1[i], p2[i])


def assert_eq_polygon(p1, p2):
    if not p1.equals(p2):
        raise ValueError


def assert_eq_multipolygon(p1, p2):
    if not p1.equals(p2):
        raise ValueError


def assert_eq_geo(geo1, geo2):
    if type(geo1) != type(geo2):
        assert TypeError
    result = geo1.equals(geo2)
    if isinstance(result, bool):
        assert result
    else:
        assert result.all()


"""
def test_geo_arrow_buffers_multipoints():
    ints = np.arange(1000)
    offsets = cudf.Series(np.random.randint(2, 10, 1000)).cumsum() * 2
    offsets = offsets[offsets < 1000]
    points = cudf.Series(ints).astype('float64')
    mpoints = [MultiPoint(points[offsets[i] : offsets[i + 1]].to_array().reshape(offsets[i] - offsets[i + 1], 2)) for i in range(len(offsets) - 1)]
    geopoints = gpd.GeoSeries(mpoints)
    cupoints = cuspatial.from_geopandas(geopoints)
    arrow = GeoArrowBuffers(mpoints_xy=points, mpoints_offsets=offsets)
    cuarrow = cuspatial.geometry.geoseries.GeoSeries(arrow)
    assert_eq(cupoints, cuarrow)


def test_geo_arrow_buffers_points():
    ints = np.arange(1000)
    series = cudf.Series(ints).astype('float64')
    points = gpd.GeoSeries(
        [Point(ints[2 * i], ints[2 * i + 1]) for i in range(500)])
    cupoints = cuspatial.from_geopandas(points)
    arrow = GeoArrowBuffers(points=series)
    cuarrow = cuspatial.geometry.geoseries.GeoSeries(arrow)
    assert_eq(cupoints, cuarrow)
"""


def test_interleaved_point(gs, polys):
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.points.x, gs[gs.type == "Point"].x.reset_index(drop=True))
    assert_eq(cugs.points.y, gs[gs.type == "Point"].y.reset_index(drop=True))
    assert_eq(
        cugs.multipoints.x,
        pd.Series(
            np.array(
                [np.array(p)[:, 0] for p in gs[gs.type == "MultiPoint"]]
            ).flatten()
        ),
    )
    assert_eq(
        cugs.multipoints.y,
        pd.Series(
            np.array(
                [np.array(p)[:, 1] for p in gs[gs.type == "MultiPoint"]]
            ).flatten()
        ),
    )
    assert_eq(
        cugs.lines.x,
        pd.Series(
            np.array([range(11, 34, 2)]).flatten(),
            dtype="float64",
        ),
    )
    assert_eq(
        cugs.lines.y,
        pd.Series(
            np.array([range(12, 35, 2)]).flatten(),
            dtype="float64",
        ),
    )
    assert_eq(cugs.polygons.x, pd.Series(polys[:, 0], dtype="float64"))
    assert_eq(cugs.polygons.y, pd.Series(polys[:, 1], dtype="float64"))


def test_to_shapely_random():
    geos_list = []
    for i in range(250):
        geo = generator(3)
        geos_list.append(geo)
    gi = gpd.GeoSeries(geos_list)
    cugs = cuspatial.from_geopandas(gi)
    cugs_back = cugs.to_geopandas()
    assert_eq_geo(gi, cugs_back)


@pytest.mark.parametrize(
    "series_slice",
    list(np.arange(10))
    + [slice(0, 12)]
    + [slice(0, 10, 1)]
    + [slice(0, 3, 1)]
    + [slice(3, 6, 1)]
    + [slice(6, 9, 1)],
)
def test_to_shapely(gs, series_slice):
    geometries = gs[series_slice]
    gi = gpd.GeoSeries(geometries)
    cugs = cuspatial.from_geopandas(gi)
    cugs_back = cugs.to_geopandas()
    assert_eq_geo(gi, cugs_back)


def test_getitem_points():
    p0 = Point([1, 2])
    p1 = Point([3, 4])
    p2 = Point([5, 6])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_point(cus[0].to_shapely(), p0)
    assert_eq_point(cus[1].to_shapely(), p1)
    assert_eq_point(cus[2].to_shapely(), p2)


def test_getitem_lines():
    p0 = LineString([[1, 2], [3, 4]])
    p1 = LineString([[1, 2], [3, 4], [5, 6], [7, 8]])
    p2 = LineString([[1, 2], [3, 4], [5, 6]])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0].to_shapely(), p0)
    assert_eq_linestring(cus[1].to_shapely(), p1)
    assert_eq_linestring(cus[2].to_shapely(), p2)


@pytest.mark.parametrize(
    "series_slice",
    list(np.arange(10))
    + [slice(0, 10, 1)]
    + [slice(0, 3, 1)]
    + [slice(3, 6, 1)]
    + [slice(6, 9, 1)],
)
def test_size(gs, series_slice):
    geometries = gs[series_slice]
    gi = gpd.GeoSeries(geometries)
    cugs = cuspatial.from_geopandas(gi)
    assert len(gi) == len(cugs)
