import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point

import cudf

from cuspatial import pairwise_point_linestring_distance
from cuspatial.io.geopandas import from_geopandas

# def print_mpoints(pts):
#     print(pts.multipoints.parts_offset)
#     print(pts.multipoints.xy)


# def print_linestring(ls):
#     print(ls.lines.parts_offset)
#     print(ls.lines.linestring_offset)
#     print(ls.lines.xy)


@pytest.mark.parametrize(
    "points, lines",
    [
        ([Point(0, 0)], [LineString([(1, 0), (0, 1)])]),
        ([MultiPoint([(0, 0), (0.5, 0.5)])], [LineString([(1, 0), (0, 1)])]),
        (
            [Point(0, 0)],
            [MultiLineString([[(1, 0), (0, 1)], [(0, 0), (1, 1)]])],
        ),
        (
            [MultiPoint([(0, 0), (1, 1)])],
            [MultiLineString([[(1, 0), (0, 1)], [(0, 0), (1, 1)]])],
        ),
    ],
)
def test_one_pair(points, lines):
    hpts = gpd.GeoSeries(points)
    hls = gpd.GeoSeries(lines)

    gpts = from_geopandas(hpts)
    gls = from_geopandas(hls)

    got = pairwise_point_linestring_distance(gpts, gls)
    expected = hpts.distance(hls)

    cudf.testing.assert_series_equal(got, cudf.Series(expected))
