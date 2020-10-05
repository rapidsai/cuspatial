# Copyright (c) 2020, NVIDIA CORPORATION.

from geopandas.geoseries import GeoSeries as gpGeoSeries

from cuspatial.geometry.geoseries import GeoSeries


def from_geoseries(geoseries):
    cugs = GeoSeries(geoseries)
    return cugs


def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.

    Possible inputs:
    GeoSeries
    """
    if isinstance(gpdf, gpGeoSeries):
        return from_geoseries(gpdf)
    print(gpdf)
