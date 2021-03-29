# Copyright (c) 2021, NVIDIA CORPORATION.

from geopandas import GeoSeries as gpGeoSeries
import numpy as np
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
)

import cudf
from cudf.tests.utils import assert_eq

from cuspatial.geometry.geocolumn import GeoColumn
from cuspatial import (
    GeoArrowBuffers,
    GeoSeries,
)


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


def test_lines():
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


def test_polygons():
    buffers = GeoArrowBuffers(
        {
            "polygons_xy": range(12),
            "polygons_polygons": np.array(range(5)),
            "polygons_rings": np.array(range(5)) * 3,
            "mpolygons": [1, 3],
        }
    )
    assert_eq(cudf.Series(range(12)), buffers.polygons.xy)
    assert len(buffers.polygons) == 3


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
