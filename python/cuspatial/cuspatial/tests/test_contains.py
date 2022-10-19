import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

import cudf

import cuspatial


def test_point_shared_with_polygon():
    point = Point([0, 0])
    polygon = Polygon([[0, 0], [0, 1], [1, 1], [0, 0]])
    point_series = cuspatial.from_geopandas(gpd.GeoSeries(point))
    polygon_series = cuspatial.from_geopandas(gpd.GeoSeries(polygon))
    result = cuspatial.point_in_polygon(
        point_series.points.x,
        point_series.points.y,
        polygon_series.polygons.ring_offset[:-1],
        polygon_series.polygons.part_offset[:-1],
        polygon_series.polygons.x,
        polygon_series.polygons.y,
    )
    cudf.testing.assert_frame_equal(result, cudf.DataFrame({0: False}))
    gpdpoint = point_series.to_pandas()
    gpdpolygon = polygon_series.to_pandas()
    pd.testing.assert_series_equal(
        gpdpolygon.contains(gpdpoint), pd.Series([False])
    )


def test_point_collinear_with_polygon():
    point = Point([0.5, 0.0])
    polygon = Polygon([[0, 0], [0, 1], [1, 1], [0, 0]])
    point_series = cuspatial.from_geopandas(gpd.GeoSeries(point))
    polygon_series = cuspatial.from_geopandas(gpd.GeoSeries(polygon))
    result = cuspatial.point_in_polygon(
        point_series.points.x,
        point_series.points.y,
        polygon_series.polygons.ring_offset[:-1],
        polygon_series.polygons.part_offset[:-1],
        polygon_series.polygons.x,
        polygon_series.polygons.y,
    )
    cudf.testing.assert_frame_equal(result, cudf.DataFrame({0: False}))
    gpdpoint = point_series.to_pandas()
    gpdpolygon = polygon_series.to_pandas()
    pd.testing.assert_series_equal(
        gpdpolygon.contains(gpdpoint), pd.Series([False])
    )
