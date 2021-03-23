# Copyright (c) 2021, NVIDIA CORPORATION.

import numpy as np

import cudf
from cudf.tests.utils import assert_eq

from cuspatial.geometry.geoarrowbuffers import GeoArrowBuffers


def test_points():
    buffers = GeoArrowBuffers({"points_xy": [0, 1, 2, 3]})
    assert_eq(cudf.Series([0, 1, 2, 3]), buffers.points.xy)
    assert len(buffers.points) == 2


def test_multipoints():
    buffers = GeoArrowBuffers(
        {
            "mpoints_xy": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "mpoints_offsets": [0, 3, 5, 8],
        }
    )
    assert_eq(cudf.Series([0, 1, 2, 3, 4, 5, 6, 7, 8]), buffers.multipoints.xy)
    assert len(buffers.multipoints) == 3


def test_lines():
    buffers = GeoArrowBuffers(
        {
            "lines_xy": range(12),
            "lines_offsets": np.array(range(5)) * 3,
            "mlines": [1, 3],
        }
    )
    assert_eq(cudf.Series(range(12)), buffers.lines.xy)
    assert len(buffers.lines) == 3


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
            "mpoints_xy": [0, 1, 2, 3, 4, 5, 6, 7, 8],
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
