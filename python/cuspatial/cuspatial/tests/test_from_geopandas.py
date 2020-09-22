# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import geopandas as gpd
import pytest
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon
)

import cudf
from cudf.tests.utils import assert_eq

import cuspatial


# data fixtures to generate complicated geopandas structs
def make_gpd():
    # make random digits and pack them into a dataframe
    # pack same digits into a series
    return gpd


def test_complex_geoseries():
    g0 = Point(1, 2)
    g1 = MultiPoint(((3, 4), (5, 6)))
    g2 = MultiPoint(((5, 6), (7, 8)))
    g3 = Point(9, 10)
    g4 = LineString(((11, 12), (13, 14)))
    g5 = MultiLineString((((15, 16), (17, 18)), ((19, 20), (21, 22))))
    g6 = MultiLineString((((23, 24), (25, 26)), ((27, 28), (29, 30))))
    g7 = LineString(((31, 32), (33, 34)))
    g8 = Polygon(
        ((35, 36), (37, 38), (39, 40), (41, 42)),
        [((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))],
    )
    g9 = MultiPolygon(
        [ (
            ((43, 44), (45, 46), (47, 48)),
            [((49, 50), (51, 52), (53, 54))],
        ),
        (
            ((55, 56), (57, 58), (59, 60)),
            [((61, 62), (63, 64), (65, 66))],
        ) ]
    )
    g10 = MultiPolygon(
        [ (
            ((67, 68), (69, 70), (71, 72)),
            [((73, 74), (75, 76), (77, 78))],
        ),
        (
            ((79, 80), (81, 82), (83, 84)),
            [
                ((85, 86), (87, 88), (89, 90)),
                ((91, 92), (93, 94), (95, 96))
            ],
        ) ]
    )
    g11 = Polygon(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
    )
    gs = gpd.GeoSeries([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
    cugs = cuspatial.from_geopandas(gs)
    assert(False)


def test_from_geopandas_point():
    gs = gpd.GeoSeries(Point(1.0, 2.0))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.points.xy, cudf.Series([1.0, 2.0]))


def test_from_geopandas_multipoint():
    gs = gpd.GeoSeries(MultiPoint([(1.0, 2.0), (3.0, 4.0)]))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.multipoints.xy, cudf.Series([1.0, 2.0, 3.0, 4.0]))
    assert_eq(cugs.multipoints.offsets, cudf.Series([0, 4]))


def test_from_geopandas_linestring():
    gs = gpd.GeoSeries(LineString(
        ((4.0, 3.0), (2.0, 1.0))
    ))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.lines.xy, cudf.Series([4.0, 3.0, 2.0, 1.0]))
    assert_eq(cugs.lines.offsets, cudf.Series([0, 4]))


def test_from_geopandas_multilinestring():
    gs = gpd.GeoSeries(
        MultiLineString(
            (
                ((1.0, 2.0), (3.0, 4.0)),
                ((5.0, 6.0), (7.0, 8.0)),
            )
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.lines.xy, cudf.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))
    assert_eq(cugs.lines.offsets, cudf.Series([4, 8]))


def test_from_geopandas_polygon():
    gs = gpd.GeoSeries(Polygon(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
    ))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.polygons.xy.exterior, cudf.Series([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    assert_eq(cugs.polygons.offsets.exterior, cudf.Series([8])) 


def test_from_geopandas_polygon_hole():
    gs = gpd.GeoSeries(
        Polygon(
            ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
            [((1.0, 1.0), (1.0, 0.0), (0.0, 0.0))],
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs.polygons.xy.exterior, cudf.Series([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    assert_eq(cugs.polygons.offsets.exterior, cudf.Series([8]))
    assert_eq(cugs.polygons.interior, cudf.Series([8]))

def test_from_geopandas_multipolygon():
    gs = gpd.GeoSeries(
        MultiPolygon(
            [ (
                ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
                [((1.0, 1.0), (1.0, 0.0), (0.0, 0.0))],
            ) ]
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs[0], cudf.Series([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    assert_eq(cugs[1], cudf.Series([8]))


def test_trajectory_distances_and_speeds_single_trajectory():
    objects, traj_offsetss = cuspatial.derive_trajectories(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],  # object_id
        [1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0],  # xs
        [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0],  # ys
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # timestamp
    )
    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsetss),
        objects["object_id"],
        objects.x,
        objects.y,
        objects.timestamp,
    )
    assert_eq(
        result.distance,
        cudf.Series([7892.922363, 6812.55908203125, 8485.28125]),
        check_names=False,
    )
    assert_eq(
        result.speed,
        cudf.Series([1973230.625, 2270853.0, 4242640.5]),
        check_names=False,
    )  # fast!


@pytest.mark.parametrize(
    "timestamp_type",
    [
        ("datetime64[ns]", 1000000000),
        ("datetime64[us]", 1000000),
        ("datetime64[ms]", 1000),
        ("datetime64[s]", 1),
    ],
)
def test_trajectory_distances_and_speeds_timestamp_types(timestamp_type):
    objects, traj_offsets = cuspatial.derive_trajectories(
        # object_id
        cudf.Series([0, 0, 1, 1]),
        # xs
        cudf.Series([0.0, 0.001, 0.0, 0.0]),  # 1 meter in x
        # ys
        cudf.Series([0.0, 0.0, 0.0, 0.001]),  # 1 meter in y
        # timestamp
        cudf.Series([0, timestamp_type[1], 0, timestamp_type[1]]).astype(
            timestamp_type[0]
        ),
    )
    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        objects.x,
        objects.y,
        objects.timestamp,
    )
    assert_eq(
        result,
        cudf.DataFrame({"distance": [1.0, 1.0], "speed": [1.0, 1.0]}),
        check_names=False,
    )
