# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import numpy as np
import pytest
from shapely.geometry import MultiLineString

import cudf

import cuspatial


def test_linestring_bounding_boxes_empty():
    result = cuspatial.linestring_bounding_boxes(
        cuspatial.GeoSeries([]),
        0,  # expansion_radius
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "minx": cudf.Series([], dtype=np.float64),
                "miny": cudf.Series([], dtype=np.float64),
                "maxx": cudf.Series([], dtype=np.float64),
                "maxy": cudf.Series([], dtype=np.float64),
            }
        ),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linestring_bounding_boxes_one(dtype):
    s = cuspatial.GeoSeries.from_linestrings_xy(
        cudf.Series(
            [5.856625, 2.488450, 5.008840, 1.333584, 4.586599, 3.460720],
            dtype=dtype,
        ),
        cudf.Series([0, 3]),
        cudf.Series([0, 1]),
    )

    result = cuspatial.linestring_bounding_boxes(s, 0)
    expected = cudf.DataFrame(s.to_geopandas().bounds, dtype=dtype)

    cudf.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_linestring_bounding_boxes_small(dtype):

    s = cuspatial.GeoSeries.from_linestrings_xy(
        cudf.Series(
            [
                2.488450,
                5.856625,
                1.333584,
                5.008840,
                3.460720,
                4.586599,
                5.039823,
                4.229242,
                5.561707,
                1.825073,
                7.103516,
                1.503906,
                7.190674,
                4.025879,
                5.998939,
                5.653384,
                5.998939,
                1.235638,
                5.573720,
                0.197808,
                6.703534,
                0.086693,
                5.998939,
                1.235638,
                2.088115,
                4.541529,
                1.034892,
                3.530299,
                2.415080,
                2.896937,
                3.208660,
                3.745936,
                2.088115,
                4.541529,
            ],
            dtype=dtype,
        ),
        cudf.Series([0, 3, 8, 12, 17]),
        cudf.Series([0, 1, 2, 3, 4]),
    )

    result = cuspatial.linestring_bounding_boxes(
        s,
        0.5,  # expansion_radius
    )
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(
            {
                "minx": cudf.Series(
                    [
                        0.8335840000000001,
                        4.5398230000000002,
                        5.0737199999999998,
                        0.53489199999999992,
                    ],
                    dtype=dtype,
                ),
                "miny": cudf.Series(
                    [
                        4.0865989999999996,
                        1.003906,
                        -0.41330699999999998,
                        2.3969369999999999,
                    ],
                    dtype=dtype,
                ),
                "maxx": cudf.Series(
                    [
                        3.9607199999999998,
                        7.6906739999999996,
                        7.2035340000000003,
                        3.7086600000000001,
                    ],
                    dtype=dtype,
                ),
                "maxy": cudf.Series(
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


def test_multilinestring_bounding_boxes_small():
    s = cuspatial.GeoSeries(
        [
            MultiLineString(
                [
                    [(0, 0), (1, 1), (2, 2), (3, 3)],
                    [(1, 0.5), (2, 1), (3, 1.5), (4, 2)],
                ]
            ),
            MultiLineString(
                [
                    [(-1, -1), (-2, -3), (-3, 4), (-8, 1)],
                    [(1, -1), (1, 6)],
                ]
            ),
        ]
    )

    result = cuspatial.linestring_bounding_boxes(s, 0.0)
    expected = cudf.DataFrame(s.to_geopandas().bounds)

    cudf.testing.assert_frame_equal(result, expected)
