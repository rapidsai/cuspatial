# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import numpy as np

import cudf

import cuspatial
from cuspatial.core.spatial.join import pip_bitmap_column_to_binary_array


def test_empty():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries([]), cuspatial.GeoSeries([])
    )
    cudf.testing.assert_frame_equal(result, cudf.DataFrame())


def test_one_point_in():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(
            cudf.Series([0.0, 0.0], dtype="f8")
        ),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series([-1, -1, 0, 1, 1, -1, -1, -1], dtype="f8"),
            cudf.Series([0, 4]),
            cudf.Series([0, 1]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: True})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_out():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(cudf.Series([1, 1], dtype="f8")),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series([-1, -1, 0, 1, 1, -1, -1, -1], dtype="f8"),
            cudf.Series([0, 4]),
            cudf.Series([0, 1]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: False})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_in_two_rings():

    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(cudf.Series([0, 0], dtype="f8")),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, -1, 1, -1, -1, 3, -1, 5, 0, 3, 1, 3, -1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )

    expected = cudf.DataFrame({0: True})
    cudf.testing.assert_frame_equal(expected, result)


# Note this test uses unclosed polygons which we don't strictly support, but
# currently work with our point-in-polygon algorithm. This may change in the
# future.
def test_one_point_in_two_unclosed_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(cudf.Series([0, 0], dtype="f8")),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, 0, 0.5, -1, 1, 3, -1, 5, 0, 4, 0.5, 3, 1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: True})
    cudf.testing.assert_frame_equal(expected, result)


def test_one_point_out_two_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(cudf.Series([1, 1], dtype="f8")),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, -1, 1, -1, -1, 3, -1, 5, 0, 3, 1, 3, -1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: False})
    cudf.testing.assert_frame_equal(expected, result)


# Note this test uses unclosed polygons which we don't strictly support, but
# currently work with our point-in-polygon algorithm. This may change in the
# future.
def test_one_point_out_two_unclosed_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(cudf.Series([1, 1], dtype="f8")),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, 0, 0.5, -1, 1, 3, -1, 5, 0, 4, 0.5, 3, 1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: False})
    cudf.testing.assert_frame_equal(expected, result)


def test_two_point_in_one_out_two_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(
            cudf.Series([0, 0, 1, 1], dtype="f8")
        ),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, -1, 1, -1, -1, 3, -1, 5, 0, 3, 1, 3, -1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: [True, False]})
    cudf.testing.assert_frame_equal(expected, result)


def test_two_point_out_one_in_two_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(
            cudf.Series([1, 1, 0, 0], dtype="f8")
        ),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, -1, 1, -1, -1, 3, -1, 5, 0, 3, 1, 3, -1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: [False, True]})
    cudf.testing.assert_frame_equal(expected, result)


def test_two_points_out_two_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(
            cudf.Series([1, 1, -1, 1], dtype="f8")
        ),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 1, 0, -1, 1, -1, -1, 3, -1, 5, 0, 3, 1, 3, -1],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: [False, False]})
    cudf.testing.assert_frame_equal(expected, result)


def test_two_points_in_two_rings():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(
            cudf.Series([0, 0, 0, 4], dtype="f8")
        ),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [-1, -1, 0, 1, 1, -1, -1, -1, -1, 3, 0, 5, 1, 3, -1, 3],
                dtype="f8",
            ),
            cudf.Series([0, 4, 8]),
            cudf.Series([0, 2]),
            cudf.Series([0, 1]),
        ),
    )
    expected = cudf.DataFrame({0: [True, True]})
    cudf.testing.assert_frame_equal(expected, result)


def test_three_points_two_features():
    result = cuspatial.point_in_polygon(
        cuspatial.GeoSeries.from_points_xy(
            cudf.Series([0, 0, -8, -8, 6.0, 6.0], dtype="f8")
        ),
        cuspatial.GeoSeries.from_polygons_xy(
            cudf.Series(
                [
                    -10.0,
                    -10.0,
                    5,
                    -10,
                    5,
                    5,
                    -10,
                    5,
                    -10,
                    -10,
                    0,
                    0,
                    10,
                    0,
                    10,
                    10,
                    0,
                    10,
                    0,
                    0,
                ],
                dtype="f8",
            ),
            cudf.Series([0, 5, 10]),
            cudf.Series([0, 1, 2]),
            cudf.Series([0, 1, 2]),
        ),
    )
    expected = cudf.DataFrame()
    expected[0] = [True, True, False]
    expected[1] = [False, False, True]
    cudf.testing.assert_frame_equal(expected, result)


def test_pip_bitmap_column_to_binary_array():
    col = cudf.Series([0b00000000, 0b00001101, 0b00000011, 0b00001001])._column
    got = pip_bitmap_column_to_binary_array(col, width=4)
    expected = np.array(
        [[0, 0, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]], dtype="int8"
    )
    np.testing.assert_array_equal(got.get(), expected)

    col = cudf.Series([], dtype="i8")._column
    got = pip_bitmap_column_to_binary_array(col, width=0)
    expected = np.array([], dtype="int8").reshape(0, 0)
    np.testing.assert_array_equal(got.get(), expected)

    col = cudf.Series([None, None], dtype="float64")._column
    got = pip_bitmap_column_to_binary_array(col, width=0)
    expected = np.array([], dtype="int8").reshape(2, 0)
    np.testing.assert_array_equal(got.get(), expected)

    col = cudf.Series(
        [
            0b000000011101110,
            0b000000000001101,
            0b111001110011010,
        ]
    )._column
    got = pip_bitmap_column_to_binary_array(col, width=15)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
        ],
        dtype="int8",
    )
    np.testing.assert_array_equal(got.get(), expected)

    col = cudf.Series([0, 0, 0])._column
    got = pip_bitmap_column_to_binary_array(col, width=3)
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype="int8")
    np.testing.assert_array_equal(got.get(), expected)
