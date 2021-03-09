# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd

from geopandas.geoseries import GeoSeries as gpGeoSeries
from geopandas import GeoDataFrame as gpGeoDataFrame

from cuspatial.geometry.geoseries import GeoSeries
from cuspatial.geometry.geodataframe import GeoDataFrame


def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.

    Possible inputs:
    pandas.Series
    geopandas.geoseries.GeoSeries
    geopandas.geodataframe.GeoDataFrame
    """
    if isinstance(gpdf, gpGeoSeries):
        return GeoSeries(gpdf)
    elif isinstance(gpdf, gpGeoDataFrame):
        return GeoDataFrame(gpdf)
    elif isinstance(gpdf, pd.Series):
        raise TypeError("Mixed pandas/geometry types not supported yet.")
    else:
        raise TypeError
