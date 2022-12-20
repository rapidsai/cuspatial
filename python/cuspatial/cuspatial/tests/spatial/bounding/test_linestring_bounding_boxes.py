# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf

import cuspatial


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linestring_bounding_boxes_empty(dtype):
    result = cuspatial.linestring_bounding_boxes(
        cudf.Series(),
        cudf.Series([], dtype=dtype),
        cudf.Series([], dtype=dtype),
        0,  # expansion_radius
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": cudf.Series([], dtype=dtype),
                "y_min": cudf.Series([], dtype=dtype),
                "x_max": cudf.Series([], dtype=dtype),
                "y_max": cudf.Series([], dtype=dtype),
            }
        ),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linestring_bounding_boxes_one(dtype):
    result = cuspatial.linestring_bounding_boxes(
        cudf.Series([0, 3]),
        cudf.Series([2.488450, 1.333584, 3.460720], dtype=dtype),
        cudf.Series([5.856625, 5.008840, 4.586599], dtype=dtype),
        0,  # expansion_radius
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": cudf.Series([1.333584], dtype=dtype),
                "y_min": cudf.Series([4.586599], dtype=dtype),
                "x_max": cudf.Series([3.460720], dtype=dtype),
                "y_max": cudf.Series([5.856625], dtype=dtype),
            }
        ),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linestring_bounding_boxes_small(dtype):
    result = cuspatial.linestring_bounding_boxes(
        cudf.Series([0, 3, 8, 12, 17]),
        cudf.Series(
            [
                # ring 1
                2.488450,
                1.333584,
                3.460720,
                # ring 2
                5.039823,
                5.561707,
                7.103516,
                7.190674,
                5.998939,
                # ring 3
                5.998939,
                5.573720,
                6.703534,
                5.998939,
                # ring 4
                2.088115,
                1.034892,
                2.415080,
                3.208660,
                2.088115,
            ],
            dtype=dtype,
        ),
        cudf.Series(
            [
                # ring 1
                5.856625,
                5.008840,
                4.586599,
                # ring 2
                4.229242,
                1.825073,
                1.503906,
                4.025879,
                5.653384,
                # ring 3
                1.235638,
                0.197808,
                0.086693,
                1.235638,
                # ring 4
                4.541529,
                3.530299,
                2.896937,
                3.745936,
                4.541529,
            ],
            dtype=dtype,
        ),
        0.5,  # expansion_radius
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "x_min": cudf.Series(
                    [
                        0.8335840000000001,
                        4.5398230000000002,
                        5.0737199999999998,
                        0.53489199999999992,
                    ],
                    dtype=dtype,
                ),
                "y_min": cudf.Series(
                    [
                        4.0865989999999996,
                        1.003906,
                        -0.41330699999999998,
                        2.3969369999999999,
                    ],
                    dtype=dtype,
                ),
                "x_max": cudf.Series(
                    [
                        3.9607199999999998,
                        7.6906739999999996,
                        7.2035340000000003,
                        3.7086600000000001,
                    ],
                    dtype=dtype,
                ),
                "y_max": cudf.Series(
                    [
                        6.3566250000000002,
                        6.153384,
                        1.735638,
                        5.0415289999999997,
                    ],
                    dtype=dtype,
                ),
            }
        ),
    )
