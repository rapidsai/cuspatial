# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import pytest
from shapely.geometry import MultiPolygon, Polygon

import cudf

import cuspatial


def test_polygon_bounding_boxes_empty():
    s = cuspatial.GeoSeries([])
    result = cuspatial.polygon_bounding_boxes(s)
    cudf.testing.assert_frame_equal(
        result,
        cudf.DataFrame(columns=["minx", "miny", "maxx", "maxy"], dtype="f8"),
    )


def test_polygon_bounding_boxes_one():
    s = cuspatial.GeoSeries(
        [
            Polygon(
                [
                    (2.488450, 5.856625),
                    (1.333584, 5.008840),
                    (3.460720, 4.586599),
                ]
            )
        ]
    )
    result = cuspatial.polygon_bounding_boxes(s)
    cudf.testing.assert_frame_equal(
        result, cudf.from_pandas(s.to_geopandas().bounds)
    )


def test_polygon_bounding_boxes_small():

    s = cuspatial.GeoSeries(
        [
            Polygon(
                [
                    (2.488450, 5.856625),
                    (1.333584, 5.008840),
                    (3.460720, 4.586599),
                ]
            ),
            Polygon(
                [
                    (5.039823, 4.229242),
                    (5.561707, 1.825073),
                    (7.103516, 1.503906),
                    (7.190674, 4.025879),
                    (5.998939, 5.653384),
                ]
            ),
            Polygon(
                [
                    (5.998939, 1.235638),
                    (5.573720, 0.197808),
                    (6.703534, 0.086693),
                    (5.998939, 1.235638),
                ]
            ),
            Polygon(
                [
                    (2.088115, 4.541529),
                    (1.034892, 3.530299),
                    (2.415080, 2.896937),
                    (3.208660, 3.745936),
                    (2.088115, 4.541529),
                ]
            ),
        ]
    )
    result = cuspatial.polygon_bounding_boxes(s)
    cudf.testing.assert_frame_equal(
        result, cudf.from_pandas(s.to_geopandas().bounds)
    )


@pytest.mark.skip(reason="MultiPolygon not yet supported")
def test_multipolygon_bounding_box():
    s = cuspatial.GeoSeries(
        [
            MultiPolygon(
                [
                    Polygon(
                        [(0, 0), (1, 0), (1, -1), (2, -1), (2, 2), (0, 0)]
                    ),
                    Polygon([(-1, -1), (-2, -2), (-3, -2), (-1, -1)]),
                ]
            ),
            MultiPolygon([Polygon([(-1, 0), (-1, 1), (0, 0), (-1, 0)])]),
        ]
    )

    expected = cuspatial.core.spatial.bounding.polygon_bounding_boxes(s)
    actual = s.to_geopandas().bounds

    cudf.testing.assert_frame_equal(expected, cudf.from_pandas(actual))
