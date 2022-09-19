# Copyright (c) 2020, NVIDIA CORPORATION.

import os

import numpy as np
import pytest

import cudf

import cuspatial
from cuspatial.io.shapefile import WindingOrder

shapefiles_path = os.path.join(
    os.environ["CUSPATIAL_HOME"], "test_fixtures", "shapefiles"
)


def test_non_existent_file():
    with pytest.raises(RuntimeError):
        f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
            "non_exist.shp"
        )


def test_zero_polygons():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "empty_poly.shp")
    )
    cudf.testing.assert_series_equal(
        f_pos, cudf.Series(dtype=np.int32, name="f_pos")
    )
    cudf.testing.assert_series_equal(
        r_pos, cudf.Series(dtype=np.int32, name="r_pos")
    )
    cudf.testing.assert_frame_equal(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series(dtype=np.float64),
                "y": cudf.Series(dtype=np.float64),
            }
        ),
    )


def test_one_polygon_reversed():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "one_poly.shp"),
        outer_ring_order=WindingOrder.CLOCKWISE,
    )
    cudf.testing.assert_series_equal(
        f_pos, cudf.Series([0], dtype=np.int32, name="f_pos")
    )
    cudf.testing.assert_series_equal(
        r_pos, cudf.Series([0], dtype=np.int32, name="r_pos")
    )
    cudf.testing.assert_frame_equal(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series([-10, 5, 5, -10, -10], dtype=np.float64),
                "y": cudf.Series([-10, -10, 5, 5, -10], dtype=np.float64),
            }
        ),
    )


def test_one_polygon():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "one_poly.shp")
    )
    cudf.testing.assert_series_equal(
        f_pos, cudf.Series([0], dtype=np.int32, name="f_pos")
    )
    cudf.testing.assert_series_equal(
        r_pos, cudf.Series([0], dtype=np.int32, name="r_pos")
    )
    cudf.testing.assert_frame_equal(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series([-10, -10, 5, 5, -10], dtype=np.float64),
                "y": cudf.Series([-10, 5, 5, -10, -10], dtype=np.float64),
            }
        ),
    )


def test_two_polygons_reversed():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "two_polys.shp"),
        outer_ring_order=WindingOrder.CLOCKWISE,
    )
    cudf.testing.assert_series_equal(
        f_pos, cudf.Series([0, 1], dtype=np.int32, name="f_pos")
    )
    cudf.testing.assert_series_equal(
        r_pos, cudf.Series([0, 5], dtype=np.int32, name="r_pos")
    )
    cudf.testing.assert_frame_equal(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series(
                    [-10, 5, 5, -10, -10, 0, 10, 10, 0, 0], dtype=np.float64
                ),
                "y": cudf.Series(
                    [-10, -10, 5, 5, -10, 0, 0, 10, 10, 0], dtype=np.float64
                ),
            }
        ),
    )


def test_two_polygons():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "two_polys.shp")
    )
    cudf.testing.assert_series_equal(
        f_pos, cudf.Series([0, 1], dtype=np.int32, name="f_pos")
    )
    cudf.testing.assert_series_equal(
        r_pos, cudf.Series([0, 5], dtype=np.int32, name="r_pos")
    )
    cudf.testing.assert_frame_equal(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series(
                    [-10, -10, 5, 5, -10, 0, 0, 10, 10, 0], dtype=np.float64
                ),
                "y": cudf.Series(
                    [-10, 5, 5, -10, -10, 0, 10, 10, 0, 0], dtype=np.float64
                ),
            }
        ),
    )
