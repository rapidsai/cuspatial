# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from enum import Enum
from numbers import Integral

import cupy as cp
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
from cudf.testing import assert_series_equal

import cuspatial
from cuspatial.testing.helpers import geometry_to_coords

np.random.seed(0)


class Example_Feature_Enum(Enum):
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
    size: Integral, has_z: bool = False, obj_type: Example_Feature_Enum = None
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


def generator(size: Integral, obj_type: Example_Feature_Enum = None):
    geos_list = []
    for i in range(size):
        geo = generate_random_shapely_feature(3, obj_type)
        geos_list.append(geo)
    return geos_list


def assert_eq_point(p1, p2):
    assert type(p1) is type(p2)
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.has_z == p2.has_z
    if p1.has_z:
        assert p1.z == p2.z
    assert True


def assert_eq_multipoint(p1, p2):
    assert type(p1) is type(p2)
    assert len(p1) == len(p2)
    for i in range(len(p1)):
        assert_eq_point(p1[i], p2[i])


def assert_eq_linestring(p1, p2):
    assert type(p1) is type(p2)
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
    if type(geo1) is not type(geo2):
        raise TypeError
    result = geo1.equals(geo2)
    if isinstance(result, bool):
        assert result
    else:
        assert result.all()


def test_interleaved_point(gs):
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

    xy, x, y = geometry_to_coords(gs, (MultiPolygon, Polygon))

    cudf.testing.assert_series_equal(
        cugs.polygons.x.reset_index(drop=True),
        cudf.Series(x, dtype="float64").reset_index(drop=True),
    )
    cudf.testing.assert_series_equal(
        cugs.polygons.y.reset_index(drop=True),
        cudf.Series(y, dtype="float64").reset_index(drop=True),
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
    gps = gpd.GeoSeries(generator(3, Example_Feature_Enum.POINT))
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


@pytest.mark.parametrize(
    "geom_access",
    [
        # Tuples: accessor, types, slice
        # slices here are meant to be supersets of the range in the gs fixture
        # that contains the types of geometries being accessed
        # Note that cuspatial.GeoSeries provides accessors for "multipoints",
        # but not for "multilinestrings" or "multipolygons"
        # (inconsistent interface)
        ("points", Point, slice(0, 6)),
        ("multipoints", MultiPoint, slice(2, 8)),
        ("lines", (LineString, MultiLineString), slice(2, 10)),
        ("polygons", (Polygon, MultiPolygon), slice(6, 12)),
    ],
)
def test_geometry_access_slicing(gs, geom_access):
    accessor, types, slice = geom_access
    xy, x, y = geometry_to_coords(gs, types)

    cugs = cuspatial.from_geopandas(gs)[slice]
    assert (getattr(cugs, accessor).x == cudf.Series(x)).all()
    assert (getattr(cugs, accessor).y == cudf.Series(y)).all()
    assert (getattr(cugs, accessor).xy == cudf.Series(xy)).all()


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


def test_memory_usage_simple(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert cugs.memory_usage() == 1616


def test_memory_usage_large(naturalearth_lowres):
    geometry = cuspatial.from_geopandas(naturalearth_lowres)["geometry"]
    # the geometry column from naturalearth_lowres is 217kb of coordinates
    assert geometry.memory_usage() == 216789


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
    expected = gps.reset_index(
        level=level, drop=drop, name=name, inplace=inplace
    )
    got = gs.reset_index(level=level, drop=drop, name=name, inplace=inplace)
    if inplace:
        expected = gps
        got = gs
    if drop:
        pd.testing.assert_series_equal(expected, got.to_pandas())
    else:
        pd.testing.assert_frame_equal(expected, got.to_pandas())


def test_geocolumn_polygon_accessor():
    s = gpd.GeoSeries(
        [
            Polygon([(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]),
            MultiPolygon(
                [
                    Polygon([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 1.0)]),
                    Polygon([(5.0, 5.0), (4.0, 4.0), (4.0, 5.0), (5.0, 5.0)]),
                ]
            ),
            Polygon(
                [(3.0, 3.0), (2.0, 3.0), (2.0, 2.0), (3.0, 2.0), (3.0, 3.0)]
            ),
        ]
    )
    gs = cuspatial.from_geopandas(s)
    expected_xy = [
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        2.0,
        1.0,
        2.0,
        2.0,
        1.0,
        1.0,
        5.0,
        5.0,
        4.0,
        4.0,
        4.0,
        5.0,
        5.0,
        5.0,
        3.0,
        3.0,
        2.0,
        3.0,
        2.0,
        2.0,
        3.0,
        2.0,
        3.0,
        3.0,
    ]

    cp.testing.assert_array_equal(gs.polygons.xy, cp.array(expected_xy))

    expected_ring_offset = [0, 4, 8, 12, 17]
    cp.testing.assert_array_equal(
        gs.polygons.ring_offset, cp.array(expected_ring_offset)
    )

    expected_part_offset = [0, 1, 2, 3, 4]
    cp.testing.assert_array_equal(
        gs.polygons.part_offset, cp.array(expected_part_offset)
    )

    expected_geometry_offset = [0, 1, 3, 4]
    cp.testing.assert_array_equal(
        gs.polygons.geometry_offset, cp.array(expected_geometry_offset)
    )

    expected_point_indices = [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
    ]
    cp.testing.assert_array_equal(
        gs.polygons.point_indices(), cp.array(expected_point_indices)
    )


def test_from_points_xy(point_generator):
    hs = gpd.GeoSeries(point_generator(10))
    gs = cuspatial.from_geopandas(hs)

    gs2 = cuspatial.GeoSeries.from_points_xy(gs.points.xy)

    gpd.testing.assert_geoseries_equal(hs, gs2.to_geopandas())


def test_from_multipoints_xy(multipoint_generator):
    hs = gpd.GeoSeries(multipoint_generator(10, max_num_geometries=10))
    gs = cuspatial.from_geopandas(hs)

    gs2 = cuspatial.GeoSeries.from_multipoints_xy(
        gs.multipoints.xy, gs.multipoints.geometry_offset
    )

    gpd.testing.assert_geoseries_equal(hs, gs2.to_geopandas())


def test_from_linestrings_xy(linestring_generator):
    hs = gpd.GeoSeries(linestring_generator(10, 10))
    gs = cuspatial.from_geopandas(hs)

    gs2 = cuspatial.GeoSeries.from_linestrings_xy(
        gs.lines.xy, gs.lines.part_offset, gs.lines.geometry_offset
    )

    gpd.testing.assert_geoseries_equal(hs, gs2.to_geopandas())


def test_from_polygons_xy(polygon_generator):
    hs = gpd.GeoSeries(polygon_generator(10, 10))
    gs = cuspatial.from_geopandas(hs)

    gs2 = cuspatial.GeoSeries.from_polygons_xy(
        gs.polygons.xy,
        gs.polygons.ring_offset,
        gs.polygons.part_offset,
        gs.polygons.geometry_offset,
    )

    gpd.testing.assert_geoseries_equal(hs, gs2.to_geopandas())


def test_from_linestrings_xy_example():
    linestrings_xy = cudf.Series([0.0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    part_offset = cudf.Series([0, 6])
    geometry_offset = cudf.Series([0, 1])
    gline = cuspatial.GeoSeries.from_linestrings_xy(
        linestrings_xy, part_offset, geometry_offset
    )
    hline = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]),
        ]
    )
    gpd.testing.assert_geoseries_equal(
        gline.to_geopandas(), hline, check_less_precise=True
    )


def test_from_polygons_xy_example():
    polygons_xy = cudf.Series([0.0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0])
    ring_offset = cudf.Series([0, 6])
    part_offset = cudf.Series([0, 1])
    geometry_offset = cudf.Series([0, 1])
    gpolygon = cuspatial.GeoSeries.from_polygons_xy(
        polygons_xy,
        ring_offset,
        part_offset,
        geometry_offset,
    )
    hpolygon = gpd.GeoSeries(
        [Polygon([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (0, 0)])]
    )
    gpd.testing.assert_geoseries_equal(gpolygon.to_geopandas(), hpolygon)


@pytest.mark.parametrize(
    "s",
    [
        gpd.GeoSeries(),
        gpd.GeoSeries([Point(0, 0)]),
        gpd.GeoSeries([None]),
        gpd.GeoSeries([Point(0, 0), None, Point(1, 1)]),
        gpd.GeoSeries([Point(0, 0), None, LineString([(1, 1), (2, 2)]), None]),
    ],
)
def test_isna(s):
    assert_series_equal(cudf.Series(s.isna()), cuspatial.GeoSeries(s).isna())


@pytest.mark.parametrize(
    "s",
    [
        gpd.GeoSeries(),
        gpd.GeoSeries([Point(0, 0)]),
        gpd.GeoSeries([None]),
        gpd.GeoSeries([Point(0, 0), None, Point(1, 1)]),
        gpd.GeoSeries([Point(0, 0), None, LineString([(1, 1), (2, 2)]), None]),
    ],
)
def test_notna(s):
    assert_series_equal(cudf.Series(s.notna()), cuspatial.GeoSeries(s).notna())
