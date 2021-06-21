# Copyright (c) 2021, NVIDIA CORPORATION.

import numpy as np
from geopandas import GeoSeries as gpGeoSeries
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf
from cudf.tests.utils import assert_eq

from cuspatial import GeoArrowBuffers, GeoSeries
from cuspatial.geometry.geocolumn import GeoColumn


def test_points():
    buffers = GeoArrowBuffers({"points_xy": [0, 1, 2, 3]})
    assert_eq(cudf.Series([0, 1, 2, 3]), buffers.points.xy)
    assert len(buffers.points) == 2
    column = GeoColumn(buffers)
    assert_eq(GeoSeries(column), gpGeoSeries([Point(0, 1), Point(2, 3)]))


def test_multipoints():
    buffers = GeoArrowBuffers(
        {"mpoints_xy": np.arange(0, 16), "mpoints_offsets": [0, 4, 8, 12, 16]}
    )
    assert_eq(cudf.Series(np.arange(0, 16)), buffers.multipoints.xy)
    assert len(buffers.multipoints) == 4
    column = GeoColumn(buffers)
    assert_eq(
        GeoSeries(column),
        gpGeoSeries(
            [
                MultiPoint([Point([0, 1]), Point([2, 3])]),
                MultiPoint([Point(4, 5), Point(6, 7)]),
                MultiPoint([Point(8, 9), Point(10, 11)]),
                MultiPoint([Point(12, 13), Point(14, 15)]),
            ]
        ),
    )


def test_homogeneous_lines():
    buffers = GeoArrowBuffers(
        {"lines_xy": range(24), "lines_offsets": np.array(range(5)) * 6}
    )
    assert_eq(cudf.Series(range(24)), buffers.lines.xy)
    assert len(buffers.lines) == 4
    column = GeoColumn(buffers)
    assert_eq(
        GeoSeries(column),
        gpGeoSeries(
            [
                LineString([[0, 1], [2, 3], [4, 5]]),
                LineString([[6, 7], [8, 9], [10, 11]]),
                LineString([[12, 13], [14, 15], [16, 17]]),
                LineString([[18, 19], [20, 21], [22, 23]]),
            ]
        ),
    )


def test_mixed_lines():
    buffers = GeoArrowBuffers(
        {
            "lines_xy": range(24),
            "lines_offsets": np.array(range(5)) * 6,
            "mlines": [1, 3],
        }
    )
    assert_eq(cudf.Series(range(24)), buffers.lines.xy)
    assert len(buffers.lines) == 3
    column = GeoColumn(buffers)
    assert_eq(
        GeoSeries(column),
        gpGeoSeries(
            [
                LineString([[0, 1], [2, 3], [4, 5]]),
                MultiLineString(
                    [
                        LineString([[6, 7], [8, 9], [10, 11]]),
                        LineString([[12, 13], [14, 15], [16, 17]]),
                    ]
                ),
                LineString([[18, 19], [20, 21], [22, 23]]),
            ]
        ),
    )


def test_homogeneous_polygons():
    polygons_xy = np.array(
        [
            np.concatenate((x[0:6], x[0:2]), axis=None)
            for x in np.arange(60).reshape(10, 6)
        ]
    )
    buffers = GeoArrowBuffers(
        {
            "polygons_xy": polygons_xy.flatten(),
            "polygons_polygons": np.array([0, 1, 3, 5, 7, 9, 10]),
            "polygons_rings": np.arange(11) * 8,
        }
    )
    assert_eq(cudf.Series(polygons_xy.flatten()), buffers.polygons.xy)
    assert len(buffers.polygons) == 6
    column = GeoColumn(buffers)
    assert_eq(
        GeoSeries(column),
        gpGeoSeries(
            [
                Polygon(((0, 1), (2, 3), (4, 5))),
                Polygon(
                    ((6, 7), (8, 9), (10, 11)),
                    [((12, 13), (14, 15), (16, 17))],
                ),
                Polygon(
                    ((18, 19), (20, 21), (22, 23)),
                    [((24, 25), (26, 27), (28, 29))],
                ),
                Polygon(
                    ((30, 31), (32, 33), (34, 35)),
                    [((36, 37), (38, 39), (40, 41))],
                ),
                Polygon(
                    ((42, 43), (44, 45), (46, 47)),
                    [((48, 49), (50, 51), (52, 53))],
                ),
                Polygon(((54, 55), (56, 57), (58, 59))),
            ]
        ),
    )


def test_polygons():
    polygons_xy = np.array(
        [
            np.concatenate((x[0:6], x[0:2]), axis=None)
            for x in np.arange(60).reshape(10, 6)
        ]
    )
    buffers = GeoArrowBuffers(
        {
            "polygons_xy": polygons_xy.flatten(),
            "polygons_polygons": np.array([0, 1, 3, 5, 7, 9, 10]),
            "polygons_rings": np.arange(11) * 8,
            "mpolygons": [2, 4],
        }
    )
    assert_eq(cudf.Series(polygons_xy.flatten()), buffers.polygons.xy)
    assert len(buffers.polygons) == 5
    column = GeoColumn(buffers)
    assert_eq(
        GeoSeries(column),
        gpGeoSeries(
            [
                Polygon(((0, 1), (2, 3), (4, 5))),
                Polygon(
                    ((6, 7), (8, 9), (10, 11)),
                    [((12, 13), (14, 15), (16, 17))],
                ),
                MultiPolygon(
                    [
                        (
                            ((18, 19), (20, 21), (22, 23)),
                            [((24, 25), (26, 27), (28, 29))],
                        ),
                        (
                            ((30, 31), (32, 33), (34, 35)),
                            [((36, 37), (38, 39), (40, 41))],
                        ),
                    ]
                ),
                Polygon(
                    ((42, 43), (44, 45), (46, 47)),
                    [((48, 49), (50, 51), (52, 53))],
                ),
                Polygon(((54, 55), (56, 57), (58, 59))),
            ]
        ),
    )


def test_full():
    buffers = GeoArrowBuffers(
        {
            "points_xy": [0, 1, 2, 3],
            "mpoints_xy": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "mpoints_offsets": [0, 3, 5, 8],
            "lines_xy": range(12),
            "lines_offsets": np.array(range(5)) * 3,
            "mlines": [1, 3],
            "polygons_xy": range(12),
            "polygons_polygons": np.array(range(5)),
            "polygons_rings": np.array(range(5)) * 3,
            "mpolygons": [1, 3],
        }
    )
    assert len(buffers) == 11
