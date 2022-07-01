# Copyright (c) 2021, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
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

from cuspatial import GeoArrowBuffers, GeoSeries
from cuspatial.geometry.geocolumn import GeoColumn


def test_points():
    buffers = GeoArrowBuffers({"points_xy": [0, 1, 2, 3]})
    cudf.testing.assert_series_equal(
        cudf.Series([0, 1, 2, 3]), buffers.points.xy
    )
    assert len(buffers.points) == 2
    column = GeoColumn(buffers)
    pd.testing.assert_series_equal(
        GeoSeries(column).to_pandas(), gpGeoSeries([Point(0, 1), Point(2, 3)])
    )


def test_multipoints():
    buffers = GeoArrowBuffers(
        {"mpoints_xy": np.arange(0, 16), "mpoints_offsets": [0, 2, 4, 6, 8]}
    )
    cudf.testing.assert_series_equal(
        cudf.Series(np.arange(0, 16)), buffers.multipoints.xy
    )
    assert len(buffers.multipoints) == 4
    column = GeoColumn(buffers)
    pd.testing.assert_series_equal(
        GeoSeries(column).to_pandas(),
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
        {"lines_xy": range(24), "lines_offsets": np.array(range(5)) * 3}
    )
    cudf.testing.assert_series_equal(cudf.Series(range(24)), buffers.lines.xy)
    assert len(buffers.lines.offsets) == 5
    column = GeoColumn(buffers)
    pd.testing.assert_series_equal(
        GeoSeries(column).to_pandas(),
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
            "lines_offsets": np.array(range(5)) * 3,
            "mlines": [0, 1, 3, 4],
        }
    )
    cudf.testing.assert_series_equal(cudf.Series(range(24)), buffers.lines.xy)
    assert len(buffers.lines.offsets) == 5
    column = GeoColumn(buffers)
    pd.testing.assert_series_equal(
        GeoSeries(column).to_pandas(),
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
    polygons_xy = np.arange(60)
    buffers = GeoArrowBuffers(
        {
            "polygons_xy": polygons_xy,
            "polygons_polygons": [0, 1, 3, 5, 7, 9, 10],
            "polygons_rings": np.arange(11) * 3,
            "mpolygons": [0, 1, 2, 3, 4, 5, 6],
        }
    )
    cudf.testing.assert_series_equal(
        cudf.Series(polygons_xy.flatten()), buffers.polygons.xy
    )
    assert len(buffers.polygons) == 6
    column = GeoColumn(buffers)
    pd.testing.assert_series_equal(
        GeoSeries(column).to_pandas(),
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
    polygons_xy = np.arange(60)
    buffers = GeoArrowBuffers(
        {
            "polygons_xy": polygons_xy,
            "polygons_polygons": [0, 1, 3, 5, 7, 9, 10],
            "polygons_rings": np.arange(11) * 3,
            "mpolygons": [0, 1, 2, 4, 5, 6],
        }
    )
    cudf.testing.assert_series_equal(
        cudf.Series(polygons_xy.flatten()), buffers.polygons.xy
    )
    assert len(buffers.polygons) == 5
    column = GeoColumn(buffers)
    pd.testing.assert_series_equal(
        GeoSeries(column).to_pandas(),
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
            "mpolygons": [0, 1, 3, 5],
        }
    )
    assert len(buffers) == 9
