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


def test_from_geopandas_point():
    gs = gpd.GeoSeries(Point(1.0, 2.0))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs[0], cudf.Series([1.0, 2.0]))


def test_from_geopandas_multipoint():
    gs = gpd.GeoSeries(MultiPoint([(1.0, 2.0), (3.0, 4.0)]))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs[0], cudf.Series([1.0, 2.0]))
    assert_eq(cugs[1], cudf.Series([2, 4]))


def test_from_geopandas_linestring():
    gs = gpd.GeoSeries(LineString(
        ((0.0, 0.0), (0.0, 0.0))
    ))
    cugs = cuspatial.from_geopandas(gs)
    assert_eq(cugs[0], cudf.Series([1.0, 2.0]))
    assert_eq(cugs[1], cudf.Series([2]))


def test_from_geopandas_multilinestring():
    single_point_dataframe = gpd.GeoSeries(
        MultiLineString(
            (
                ((0.0, 0.0), (0.0, 0.0)),
                ((0.0, 0.0), (0.0, 0.0)),
            )
        )
    )


def test_from_geopandas_polygon():
    single_point_dataframe = gpd.GeoSeries(Polygon(
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    ))


def test_from_geopandas_multipolygon():
    single_point_dataframe = gpd.GeoSeries(
        MultiPolygon(
            [ (
                ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
                [((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))],
            ) ]
        )
    )


def test_trajectory_distances_and_speeds_single_trajectory():
    objects, traj_offsets = cuspatial.derive_trajectories(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],  # object_id
        [1.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 3.0, 6.0, 0.0, 3.0, 6.0],  # xs
        [0.0, 1.0, 2.0, 3.0, 1.0, 3.0, 5.0, 6.0, 5.0, 4.0, 7.0, 4.0],  # ys
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # timestamp
    )
    result = cuspatial.trajectory_distances_and_speeds(
        len(traj_offsets),
        objects["object_id"],
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(
        result["distance"],
        cudf.Series([7892.922363, 6812.55908203125, 8485.28125]),
        check_names=False,
    )
    assert_eq(
        result["speed"],
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
        objects["x"],
        objects["y"],
        objects["timestamp"],
    )
    assert_eq(
        result,
        cudf.DataFrame({"distance": [1.0, 1.0], "speed": [1.0, 1.0]}),
        check_names=False,
    )
