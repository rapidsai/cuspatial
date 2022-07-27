# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.affinity import rotate
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf

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
    assert p1 == p2


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


def test_interleaved_point(gs, polys):
    cugs = cuspatial.from_geopandas(gs)
    pd.testing.assert_series_equal(
        cugs.points.x.to_pandas(),
        gs[gs.type == "Point"].x,
        check_index=False,
    )
    pd.testing.assert_series_equal(
        cugs.points.y.to_pandas(),
        gs[gs.type == "Point"].y,
        check_index=False,
    )
    cudf.testing.assert_series_equal(
        cugs.multipoints.x.reset_index(drop=True),
        cudf.Series(
            np.array(
                [
                    np.array(p.__geo_interface__["coordinates"])[:, 0]
                    for p in gs[gs.type == "MultiPoint"]
                ]
            ).flatten()
        ).reset_index(drop=True),
    )
    cudf.testing.assert_series_equal(
        cugs.multipoints.y.reset_index(drop=True),
        cudf.Series(
            np.array(
                [
                    np.array(p.__geo_interface__["coordinates"])[:, 1]
                    for p in gs[gs.type == "MultiPoint"]
                ]
            ).flatten()
        ).reset_index(drop=True),
    )
    cudf.testing.assert_series_equal(
        cugs.lines.x.reset_index(drop=True),
        cudf.Series(
            np.array([range(11, 34, 2)]).flatten(),
            dtype="float64",
        ).reset_index(drop=True),
    )
    cudf.testing.assert_series_equal(
        cugs.lines.y.reset_index(drop=True),
        cudf.Series(
            np.array([range(12, 35, 2)]).flatten(),
            dtype="float64",
        ).reset_index(drop=True),
    )
    cudf.testing.assert_series_equal(
        cugs.polygons.x.reset_index(drop=True),
        cudf.Series(polys[:, 0], dtype="float64").reset_index(drop=True),
    )
    cudf.testing.assert_series_equal(
        cugs.polygons.y.reset_index(drop=True),
        cudf.Series(polys[:, 1], dtype="float64").reset_index(drop=True),
    )


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
    "pre_slice",
    [
        list(np.arange(10)),
        (slice(0, 12)),
        (slice(0, 10, 1)),
        (slice(0, 3, 1)),
        (slice(3, 6, 1)),
        (slice(6, 9, 1)),
    ],
)
def test_to_shapely(gs, pre_slice):
    geometries = gs[pre_slice]
    gi = gpd.GeoSeries(geometries)
    cugs = cuspatial.from_geopandas(gi)
    cugs_back = cugs.to_geopandas()
    assert_eq_geo(gi, cugs_back)


@pytest.mark.parametrize(
    "series_boolmask",
    [
        np.repeat(True, 12),
        np.repeat((np.repeat(True, 3), np.repeat(False, 3)), 2).flatten(),
        np.repeat(False, 12),
        np.repeat((np.repeat(False, 3), np.repeat(True, 3)), 2).flatten(),
        np.repeat([True, False], 6).flatten(),
    ],
)
def test_boolmask(gs, series_boolmask):
    gi = gpd.GeoSeries(gs)
    cugs = cuspatial.from_geopandas(gi)
    cugs_back = cugs.to_geopandas()
    assert_eq_geo(gi[series_boolmask], cugs_back[series_boolmask])


def test_getitem_points():
    p0 = Point([1, 2])
    p1 = Point([3, 4])
    p2 = Point([5, 6])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps).to_pandas()
    assert_eq_point(cus[0], p0)
    assert_eq_point(cus[1], p1)
    assert_eq_point(cus[2], p2)


def test_getitem_lines():
    p0 = LineString([[1, 2], [3, 4]])
    p1 = LineString([[1, 2], [3, 4], [5, 6], [7, 8]])
    p2 = LineString([[1, 2], [3, 4], [5, 6]])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps).to_pandas()
    assert_eq_linestring(cus[0], p0)
    assert_eq_linestring(cus[1], p1)
    assert_eq_linestring(cus[2], p2)


def test_getitem_slice_points():
    p0 = Point([1, 2])
    p1 = Point([3, 4])
    p2 = Point([5, 6])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_point(cus[0:1][0].to_shapely(), gps[0:1][0])
    assert_eq_point(cus[0:2][0].to_shapely(), gps[0:2][0])
    assert_eq_point(cus[1:2][1].to_shapely(), gps[1:2][1])
    assert_eq_point(cus[0:3][0].to_shapely(), gps[0:3][0])
    assert_eq_point(cus[1:3][1].to_shapely(), gps[1:3][1])
    assert_eq_point(cus[2:3][2].to_shapely(), gps[2:3][2])


def test_getitem_slice_lines():
    p0 = LineString([[1, 2], [3, 4]])
    p1 = LineString([[1, 2], [3, 4], [5, 6], [7, 8]])
    p2 = LineString([[1, 2], [3, 4], [5, 6]])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0:1][0].to_shapely(), gps[0:1][0])
    assert_eq_linestring(cus[0:2][0].to_shapely(), gps[0:2][0])
    assert_eq_linestring(cus[1:2][1].to_shapely(), gps[1:2][1])
    assert_eq_linestring(cus[0:3][0].to_shapely(), gps[0:3][0])
    assert_eq_linestring(cus[1:3][1].to_shapely(), gps[1:3][1])
    assert_eq_linestring(cus[2:3][2].to_shapely(), gps[2:3][2])


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


def test_memory_usage(gs):
    assert gs.memory_usage() == 224
