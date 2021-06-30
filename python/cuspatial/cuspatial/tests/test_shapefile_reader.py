# Copyright (c) 2020, NVIDIA CORPORATION.

import os

import numpy as np
import pytest

import cudf
from cudf.testing._utils import assert_eq

import cuspatial

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
    assert_eq(f_pos, cudf.Series(dtype=np.int32, name="f_pos"))
    assert_eq(r_pos, cudf.Series(dtype=np.int32, name="r_pos"))
    assert_eq(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series(dtype=np.float64),
                "y": cudf.Series(dtype=np.float64),
            }
        ),
    )


def test_one_polygon():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "one_poly.shp")
    )
    assert_eq(f_pos, cudf.Series([0], dtype=np.int32, name="f_pos"))
    assert_eq(r_pos, cudf.Series([0], dtype=np.int32, name="r_pos"))
    assert_eq(
        points,
        cudf.DataFrame(
            {
                "x": cudf.Series([-10, 5, 5, -10, -10], dtype=np.float64),
                "y": cudf.Series([-10, -10, 5, 5, -10], dtype=np.float64),
            }
        ),
    )


def test_two_polygons():
    f_pos, r_pos, points = cuspatial.read_polygon_shapefile(
        os.path.join(shapefiles_path, "two_polys.shp")
    )
    assert_eq(f_pos, cudf.Series([0, 1], dtype=np.int32, name="f_pos"))
    assert_eq(r_pos, cudf.Series([0, 5], dtype=np.int32, name="r_pos"))
    assert_eq(
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
