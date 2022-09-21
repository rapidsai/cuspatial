import pytest

from shapely.geometry import Point, MultiPoint, LineString, MultiLineString

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

import cudf
import cuspatial

def test_empty_input():
    points = cuspatial.GeoSeries([])
    linestrings = cuspatial.GeoSeries([])

    result = cuspatial.pairwise_point_linestring_nearest_points(points, linestrings)
    expected = gpd.GeoDataFrame({
        "point_geometry_id": [],
        "linestring_geometry_id": [],
        "segment_id": [],
        "geometry": gpd.GeoSeries(),
    })
    assert_geodataframe_equal(result.to_pandas(), expected, check_index_type=False)

def test_single_pair():
    points = cuspatial.GeoSeries([Point(0, 0)])
    linestrings = cuspatial.GeoSeries([LineString([(2, 2), (1, 1)])])

    result = cuspatial.pairwise_point_linestring_nearest_points(points, linestrings)
    expected = gpd.GeoDataFrame({
        "point_geometry_id": [0],
        "linestring_geometry_id": [0],
        "segment_id": [0],
        "geometry": gpd.GeoSeries([Point(1, 1)]),
    })
    assert_geodataframe_equal(result.to_pandas(), expected, check_index_type=False)
