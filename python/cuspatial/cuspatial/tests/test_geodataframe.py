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
from cudf.testing import assert_series_equal

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


def assert_eq_polygon(p1, p2):
    if not p1.equals(p2):
        raise ValueError


def assert_eq_multipolygon(p1, p2):
    if not p1.equals(p2):
        raise ValueError


def assert_eq_geo_df(geo1, geo2):
    if type(geo1) != type(geo2):
        assert TypeError
    assert geo1.columns.equals(geo2.columns)
    for col in geo1.columns:
        assert geo1[col].equals(geo2[col])


def test_select_multiple_columns(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    pd.testing.assert_frame_equal(cugpdf[["geometry", "key"]].to_pandas(), gpdf[["geometry", "key"]])


def test_sort_values(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    sort_gpdf = gpdf.sort_values("random")
    sort_cugpdf = cugpdf.sort_values("random").to_pandas()
    pd.testing.assert_frame_equal(sort_gpdf, sort_cugpdf)


def test_groupby(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    pd.testing.assert_frame_equal(
        gpdf.groupby("key").min().sort_index(),
        cugpdf.groupby("key").min().sort_index().to_pandas(),
    )


def test_type_persistence(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    assert type(cugpdf["geometry"]) == cuspatial.geometry.geoseries.GeoSeries


def test_interleaved_point(gpdf, polys):
    cugpdf = cuspatial.from_geopandas(gpdf)
    cugs = cugpdf["geometry"]
    gs = gpdf["geometry"]
    pd.testing.assert_series_equal(cugs.points.x.to_pandas(), gs[gs.type == "Point"].x.reset_index(drop=True))
    pd.testing.assert_series_equal(cugs.points.y.to_pandas(), gs[gs.type == "Point"].y.reset_index(drop=True))
    assert_series_equal(
        cugs.multipoints.x,
        cudf.Series(
            np.array(
                [np.array(p)[:, 0] for p in gs[gs.type == "MultiPoint"]]
            ).flatten()
        ),
    )
    assert_series_equal(
        cugs.multipoints.y,
        cudf.Series(
            np.array(
                [np.array(p)[:, 1] for p in gs[gs.type == "MultiPoint"]]
            ).flatten()
        ),
    )
    assert_series_equal(
        cugs.lines.x,
        cudf.Series(np.array([range(11, 34, 2)]).flatten(), dtype="float64",),
    )
    assert_series_equal(
        cugs.lines.y,
        cudf.Series(np.array([range(12, 35, 2)]).flatten(), dtype="float64",),
    )
    assert_series_equal(cugs.polygons.x, cudf.Series(polys[:, 0], dtype="float64"))
    assert_series_equal(cugs.polygons.y, cudf.Series(polys[:, 1], dtype="float64"))


def test_to_shapely_random():
    geos_list = []
    for i in range(250):
        geo = generator(3)
        geos_list.append(geo)
    gi = gpd.GeoDataFrame(
        {"geometry": geos_list, "integer": range(len(geos_list))}
    )
    cugpdf = cuspatial.from_geopandas(gi)
    cugpdf_back = cugpdf.to_geopandas()
    assert_eq_geo_df(gi, cugpdf_back)


@pytest.mark.parametrize(
    "series_slice",
    [slice(0, 12)]
    + [slice(0, 10, 1)]
    + [slice(0, 3, 1)]
    + [slice(3, 6, 1)]
    + [slice(6, 9, 1)],
)
def test_to_shapely(gpdf, series_slice):
    geometries = gpdf.iloc[series_slice, :]
    gi = gpd.GeoDataFrame(geometries)
    cugpdf = cuspatial.from_geopandas(gi)
    cugpdf_back = cugpdf.to_geopandas()
    assert_eq_geo_df(gi, cugpdf_back)
