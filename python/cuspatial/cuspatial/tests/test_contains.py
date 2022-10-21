import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

import cuspatial


def test_point_shared_with_polygon():
    point_series = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                Point([0.25, 0.5]),
                Point([1, 1]),
                Point([0.5, 0.25]),
            ]
        )
    )
    polygon_series = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                Polygon([[0, 0], [0, 1], [1, 1], [0, 0]]),
                Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
                Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            ]
        )
    )
    gpdpoint = point_series.to_pandas()
    gpdpolygon = polygon_series.to_pandas()
    pd.testing.assert_series_equal(
        gpdpolygon.contains(gpdpoint),
        polygon_series.contains(point_series).to_pandas(),
    )


def test_point_collinear_with_polygon():
    point = Point([0.5, 0.0])
    polygon = Polygon([[0, 0], [0, 1], [1, 1], [0, 0]])
    point_series = cuspatial.from_geopandas(gpd.GeoSeries(point))
    polygon_series = cuspatial.from_geopandas(gpd.GeoSeries(polygon))
    gpdpoint = point_series.to_pandas()
    gpdpolygon = polygon_series.to_pandas()
    pd.testing.assert_series_equal(
        gpdpolygon.contains(gpdpoint),
        polygon_series.contains(point_series).to_pandas(),
    )
