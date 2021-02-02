# Copyright (c) 2020, NVIDIA CORPORATION.

from geopandas.geoseries import GeoSeries as gpGeoSeries
from geopandas import GeoDataFrame as gpGeoDataFrame

from cuspatial.geometry.geoseries import GeoSeries
from cuspatial.geometry.geodataframe import GeoDataFrame


def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.

    Possible inputs:
    GeoSeries
    GeoDataFrame
    """
    if isinstance(gpdf, gpGeoSeries):
        return GeoSeries(gpdf)
    if isinstance(gpdf, gpGeoDataFrame):
        return GeoDataFrame(gpdf)
