# Copyright (c) 2020, NVIDIA CORPORATION.

from geopandas.geoseries import GeoSeries as gpGeoSeries

from cuspatial.geometry.geoseries import GeoSeries


def from_geoseries(geoseries):
    # do in parallel: compute offsets of the geoseries
    # do in parallel: accumulate the size of the geoseries
    # allocate memory for the full size
    # do in parallel: copy each geoseries into the proper offset
    # return cuSeries
    # offsets?
    cugs = GeoSeries(geoseries)
    return cugs


def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.

    Possible inputs:
    GeoSeries
    GeoDataframe
    Point
    MultiPoint
    LineString
    MultiLineString
    Polygon
    MultiPolygon
    """
    if isinstance(gpdf, gpGeoSeries):
        return from_geoseries(gpdf)
    print(gpdf)
