# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from enum import Enum
from numbers import Integral

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geoseries_equal
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
from cuspatial.io.shapefile import WindingOrder

np.random.seed(0)


class Test_Feature_Enum(Enum):
    POINT = 0
    MULTIPOINT = 1
    LINESTRING = 2
    POLYGON = 3


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


def generate_random_shapely_feature(
    size: Integral, has_z: bool = False, obj_type: Test_Feature_Enum = None
):
    obj_type = obj_type.value if obj_type else np.random.randint(1, 7)
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


def generator(size: Integral, obj_type: Test_Feature_Enum = None):
    geos_list = []
    for i in range(size):
        geo = generate_random_shapely_feature(3, obj_type)
        geos_list.append(geo)
    return geos_list


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
    geos_list = generator(250)
    gi = gpd.GeoSeries(geos_list)
    cugs = cuspatial.from_geopandas(gi)
    cugs_back = cugs.to_geopandas()
    assert_eq_geo(gi, cugs_back)


@pytest.mark.parametrize(
    "pre_slice",
    [
        list(np.arange(10)),
        slice(0, 12),
        slice(0, 10, 1),
        slice(0, 3, 1),
        slice(3, 6, 1),
        slice(6, 9, 1),
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
    assert_eq_geo(gi[series_boolmask], cugs[series_boolmask].to_geopandas())


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


def test_getitem_slice_same_index():
    gps = gpd.GeoSeries(generator(3, Test_Feature_Enum.POINT))
    cus = cuspatial.from_geopandas(gps)
    assert_eq_geo(cus[0:1].to_geopandas(), gps[0:1])
    assert_eq_geo(cus[0:1].to_geopandas(), gps[0:1])
    assert_eq_geo(cus[0:1].to_geopandas(), gps[0:1])
    assert_eq_geo(cus[0:3].to_geopandas(), gps[0:3])
    assert_eq_geo(cus[1:3].to_geopandas(), gps[1:3])
    assert_eq_geo(cus[2:3].to_geopandas(), gps[2:3])


def test_getitem_slice_points_loc():
    p0 = Point([1, 2])
    p1 = Point([3, 4])
    p2 = Point([5, 6])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_point(cus[0:1][0], gps[0:1][0])
    assert_eq_point(cus[0:2][0], gps[0:2][0])
    assert_eq_point(cus[1:2][1], gps[1:2][1])
    assert_eq_point(cus[0:3][0], gps[0:3][0])
    assert_eq_point(cus[1:3][1], gps[1:3][1])
    assert_eq_point(cus[2:3][2], gps[2:3][2])


def test_getitem_slice_points():
    p0 = Point([1, 2])
    p1 = Point([3, 4])
    p2 = Point([5, 6])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_point(cus[0:1].iloc[0], gps[0:1].iloc[0])
    assert_eq_point(cus[0:2].iloc[0], gps[0:2].iloc[0])
    assert_eq_point(cus[1:2].iloc[0], gps[1:2].iloc[0])
    assert_eq_point(cus[0:3].iloc[0], gps[0:3].iloc[0])
    assert_eq_point(cus[1:3].iloc[0], gps[1:3].iloc[0])
    assert_eq_point(cus[2:3].iloc[0], gps[2:3].iloc[0])


def test_getitem_slice_lines():
    p0 = LineString([[1, 2], [3, 4]])
    p1 = LineString([[1, 2], [3, 4], [5, 6], [7, 8]])
    p2 = LineString([[1, 2], [3, 4], [5, 6]])
    gps = gpd.GeoSeries([p0, p1, p2])
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0:1].iloc[0], gps[0:1].iloc[0])
    assert_eq_linestring(cus[0:2].iloc[0], gps[0:2].iloc[0])
    assert_eq_linestring(cus[1:2].iloc[0], gps[1:2].iloc[0])
    assert_eq_linestring(cus[0:3].iloc[0], gps[0:3].iloc[0])
    assert_eq_linestring(cus[1:3].iloc[0], gps[1:3].iloc[0])
    assert_eq_linestring(cus[2:3].iloc[0], gps[2:3].iloc[0])


def test_getitem_slice_mlines(gs):
    gps = gs[gs.type == "MultiLineString"]
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0:1].iloc[0], gps[0:1].iloc[0])
    assert_eq_linestring(cus[0:2].iloc[0], gps[0:2].iloc[0])
    assert_eq_linestring(cus[1:2].iloc[0], gps[1:2].iloc[0])


def test_getitem_slice_polygons(gs):
    gps = gs[gs.type == "Polygon"]
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0:1].iloc[0], gps[0:1].iloc[0])
    assert_eq_linestring(cus[0:2].iloc[0], gps[0:2].iloc[0])
    assert_eq_linestring(cus[1:2].iloc[0], gps[1:2].iloc[0])


def test_getitem_slice_mpolygons(gs):
    gps = gs[gs.type == "MultiPolygon"]
    cus = cuspatial.from_geopandas(gps)
    assert_eq_linestring(cus[0:1].iloc[0], gps[0:1].iloc[0])
    assert_eq_linestring(cus[0:2].iloc[0], gps[0:2].iloc[0])
    assert_eq_linestring(cus[1:2].iloc[0], gps[1:2].iloc[0])


@pytest.mark.parametrize(
    "series_slice",
    [
        list(np.arange(10)),
        slice(0, 10, 1),
        slice(0, 3, 1),
        slice(3, 6, 1),
        slice(6, 9, 1),
    ],
)
def test_size(gs, series_slice):
    geometries = gs[series_slice]
    gi = gpd.GeoSeries(geometries)
    cugs = cuspatial.from_geopandas(gi)
    assert len(gi) == len(cugs)


def test_geometry_point_slicing(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert (cugs[:1].points.x == cudf.Series([-1])).all()
    assert (cugs[:1].points.y == cudf.Series([0])).all()
    assert (cugs[:1].points.xy == cudf.Series([-1, 0])).all()
    assert (cugs[3:].points.x == cudf.Series([9])).all()
    assert (cugs[3:].points.y == cudf.Series([10])).all()
    assert (cugs[3:].points.xy == cudf.Series([9, 10])).all()
    assert (cugs[0:4].points.x == cudf.Series([-1, 9])).all()
    assert (cugs[0:4].points.y == cudf.Series([0, 10])).all()
    assert (cugs[0:4].points.xy == cudf.Series([-1, 0, 9, 10])).all()


def test_geometry_multipoint_slicing(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert (cugs[:2].multipoints.x == cudf.Series([1, 3])).all()
    assert (cugs[:2].multipoints.y == cudf.Series([2, 4])).all()
    assert (cugs[:2].multipoints.xy == cudf.Series([1, 2, 3, 4])).all()
    assert (cugs[2:].multipoints.x == cudf.Series([5, 7])).all()
    assert (cugs[2:].multipoints.y == cudf.Series([6, 8])).all()
    assert (cugs[2:].multipoints.xy == cudf.Series([5, 6, 7, 8])).all()
    assert (cugs[0:4].multipoints.x == cudf.Series([1, 3, 5, 7])).all()
    assert (cugs[0:4].multipoints.y == cudf.Series([2, 4, 6, 8])).all()
    assert (
        cugs[0:4].multipoints.xy == cudf.Series([1, 2, 3, 4, 5, 6, 7, 8])
    ).all()


def test_geometry_linestring_slicing(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert (cugs[:5].lines.x == cudf.Series([11, 13])).all()
    assert (cugs[:5].lines.y == cudf.Series([12, 14])).all()
    assert (cugs[:5].lines.xy == cudf.Series([11, 12, 13, 14])).all()
    assert (cugs[:6].lines.x == cudf.Series([11, 13, 15, 17, 19, 21])).all()
    assert (cugs[:6].lines.y == cudf.Series([12, 14, 16, 18, 20, 22])).all()
    assert (
        cugs[:6].lines.xy
        == cudf.Series([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    ).all()
    assert (cugs[7:].lines.x == cudf.Series([31, 33])).all()
    assert (cugs[7:].lines.y == cudf.Series([32, 34])).all()
    assert (cugs[7:].lines.xy == cudf.Series([31, 32, 33, 34])).all()
    assert (cugs[6:].lines.x == cudf.Series([23, 25, 27, 29, 31, 33])).all()
    assert (cugs[6:].lines.y == cudf.Series([24, 26, 28, 30, 32, 34])).all()
    assert (
        cugs[6:].lines.xy
        == cudf.Series([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])
    ).all()


def test_geometry_polygon_slicing(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert (cugs[:9].polygons.x == cudf.Series([35, 37, 39, 41, 35])).all()
    assert (cugs[:9].polygons.y == cudf.Series([36, 38, 40, 42, 36])).all()
    assert (
        cugs[:9].polygons.xy
        == cudf.Series([35, 36, 37, 38, 39, 40, 41, 42, 35, 36])
    ).all()
    assert (
        cugs[:10].polygons.x
        == cudf.Series(
            [
                35,
                37,
                39,
                41,
                35,
                43,
                45,
                47,
                43,
                49,
                51,
                53,
                49,
                55,
                57,
                59,
                55,
                61,
                63,
                65,
                61,
            ]
        )
    ).all()
    assert (
        cugs[:10].polygons.y
        == cudf.Series(
            [
                36,
                38,
                40,
                42,
                36,
                44,
                46,
                48,
                44,
                50,
                52,
                54,
                50,
                56,
                58,
                60,
                56,
                62,
                64,
                66,
                62,
            ]
        )
    ).all()
    assert (
        cugs[11:].polygons.x
        == cudf.Series([97, 99, 102, 101, 97, 106, 108, 110, 113, 106])
    ).all()
    assert (
        cugs[11:].polygons.y
        == cudf.Series([98, 101, 103, 108, 98, 107, 109, 111, 108, 107])
    ).all()
    assert (
        cugs[11:].polygons.xy
        == cudf.Series(
            [
                97,
                98,
                99,
                101,
                102,
                103,
                101,
                108,
                97,
                98,
                106,
                107,
                108,
                109,
                110,
                111,
                113,
                108,
                106,
                107,
            ]
        )
    ).all()


def test_loc(gs):
    index = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    gs.index = index
    cugs = cuspatial.from_geopandas(gs)
    gsslice = gs[0:5]
    cugsslice = cugs[0:5]
    assert_eq_geo(gsslice, cugsslice.to_geopandas())
    gsslice = gs[["l", "k", "j", "i"]]
    cugsslice = cugs[["l", "k", "j", "i"]]
    assert_eq_geo(gsslice, cugsslice.to_geopandas())


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [
            Point(1, 1),
            MultiPoint([(2, 2), (3, 3)]),
            Point(4, 4),
            Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
        ],
        gpd.GeoSeries(
            [
                MultiLineString(
                    [[(0, 0), (1, 2), (1, 0)], [(-1, -1), (-1, 3), (0, 0)]]
                ),
                MultiPolygon(
                    [
                        Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
                        Polygon([(2, 2), (2, 3), (3, 3), (2, 2)]),
                    ]
                ),
            ]
        ),
    ],
)
def test_construction_from_foreign_object(data):
    cugs = cuspatial.GeoSeries(data)
    gps = gpd.GeoSeries(data)

    assert_geoseries_equal(cugs.to_geopandas(), gps)


def test_shapefile_constructor():
    host_dataframe = gpd.read_file(
        gpd.datasets.get_path("naturalearth_lowres")
    )
    # Shapefile reader only works with Polygons so we drop
    # all the MultiPolygons by slicing the first 19 out.
    gs = host_dataframe["geometry"][
        host_dataframe["geometry"].type == "Polygon"
    ][19:]
    gs.to_file("naturalearth_lowres_polygon")
    data = cuspatial.read_polygon_shapefile("naturalearth_lowres_polygon")
    cus = cuspatial.GeoSeries(data)

    assert_eq_geo(gs.reset_index(drop=True), cus.to_geopandas())


def test_shapefile_constructor_reversed():
    host_dataframe = gpd.read_file(
        gpd.datasets.get_path("naturalearth_lowres")
    )
    # Shapefile reader only works with Polygons so we drop
    # all the MultiPolygons by slicing the first 19 out.
    gs = host_dataframe["geometry"][
        host_dataframe["geometry"].type == "Polygon"
    ][19:]
    gs.to_file("naturalearth_lowres_polygon")
    data = cuspatial.read_polygon_shapefile(
        "naturalearth_lowres_polygon", outer_ring_order=WindingOrder.CLOCKWISE
    )
    cus = cuspatial.GeoSeries(data)

    from shapely.geometry import polygon

    reversed_gs = gpd.GeoSeries([polygon.orient(p, 1) for p in gs])
    assert_eq_geo(reversed_gs, cus.to_geopandas())


def test_memory_usage_simple(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert cugs.memory_usage() == 1616


def test_memory_usage_large():
    host_dataframe = gpd.read_file(
        gpd.datasets.get_path("naturalearth_lowres")
    )
    geometry = cuspatial.from_geopandas(host_dataframe)["geometry"]
    # the geometry column from naturalearth_lowres is 217kb of coordinates
    assert geometry.memory_usage() == 216793


@pytest.mark.parametrize("level", [None, 0, 1])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index(level, drop, name, inplace):
    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    gps = gpd.GeoSeries(
        [Point(0, 0), Point(0, 1), Point(2, 2), Point(3, 3)], index=midx
    )
    gs = cuspatial.from_geopandas(gps)
    expected = gps.reset_index(level, drop, name, inplace)
    got = gs.reset_index(level, drop, name, inplace)
    if inplace:
        expected = gps
        got = gs
    if drop:
        pd.testing.assert_series_equal(expected, got.to_pandas())
    else:
        pd.testing.assert_frame_equal(expected, got.to_pandas())
