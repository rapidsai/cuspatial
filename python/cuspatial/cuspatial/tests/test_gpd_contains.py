
import pytest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

import cudf

import cuspatial


@pytest.mark.parametrize(
    "point, polygon, expects", 
    [
        # wound clockwise, should be false
        [
            Point([0, 0]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([0.0, 1.0]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([1.0, 1.0]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([0.0, 0.5]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([0.5, 0.5]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([0.5, 1]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            False
        ],
        # wound clockwise, should be true
        [
            Point([0.25, 0.5]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            True
        ],
        [
            Point([0.75, 0.9]),
            Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
            True
        ],
        # wound counter clockwise, should be false
        [
            Point([0.0, 0.0]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([1.0, 0.0]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([1.0, 1.0]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([0.5, 0.0]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            False
        ],
        [
            Point([0.5, 0.5]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            False
        ],
        # wound counter clockwise, should be true
        [
            Point([0.5, 0.25]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            True
        ],
        [
            Point([0.9, 0.75]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            True
        ],
    ]
)
def test_point_in_polygon(point, polygon, expects):
    point_series = cuspatial.from_geopandas(gpd.GeoSeries(point))
    polygon_series = cuspatial.from_geopandas(gpd.GeoSeries(polygon))
    result = cuspatial.point_in_polygon(
        point_series.points.x,
        point_series.points.y,
        polygon_series.polygons.part_offset[:-1],
        polygon_series.polygons.ring_offset[:-1],
        polygon_series.polygons.x,
        polygon_series.polygons.y,
    )
    result[0].name = None
    gpdpoint = point_series.to_pandas()
    gpdpolygon = polygon_series.to_pandas()
    assert gpdpolygon.contains(gpdpoint).values == result[0].values_host
