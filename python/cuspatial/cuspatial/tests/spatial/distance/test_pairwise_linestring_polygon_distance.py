import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

import cudf
from cudf.testing import assert_series_equal

import cuspatial


def test_linestring_polygon_empty():
    lhs = cuspatial.GeoSeries.from_linestrings_xy([], [0], [0])
    rhs = cuspatial.GeoSeries.from_polygons_xy([], [0], [0], [0])

    got = cuspatial.pairwise_linestring_polygon_distance(lhs, rhs)

    expect = cudf.Series([], dtype="f8")

    assert_series_equal(got, expect)


@pytest.mark.parametrize(
    "linestrings",
    [
        [LineString([(0, 0), (1, 1)])],
        [MultiLineString([[(1, 1), (2, 2)], [(10, 10), (11, 11)]])],
    ],
)
@pytest.mark.parametrize(
    "polygons",
    [
        [Polygon([(0, 1), (1, 0), (-1, 0), (0, 1)])],
        [
            MultiPolygon(
                [
                    Polygon([(-2, 0), (-1, 0), (-1, -1), (-2, 0)]),
                    Polygon([(1, 0), (2, 0), (1, -1), (1, 0)]),
                ]
            )
        ],
    ],
)
def test_one_pair(linestrings, polygons):
    lhs = gpd.GeoSeries(linestrings)
    rhs = gpd.GeoSeries(polygons)

    dlhs = cuspatial.GeoSeries(linestrings)
    drhs = cuspatial.GeoSeries(polygons)

    expect = lhs.distance(rhs)
    got = cuspatial.pairwise_linestring_polygon_distance(dlhs, drhs)

    assert_series_equal(got, cudf.Series(expect))
