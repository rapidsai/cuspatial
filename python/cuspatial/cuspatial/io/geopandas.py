# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import pandas as pd
from geopandas import GeoDataFrame as gpGeoDataFrame
from geopandas.geoseries import GeoSeries as gpGeoSeries

from cuspatial.geometry.geodataframe import GeoDataFrame
from cuspatial.geometry.geoseries import GeoSeries


def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.

    Possible inputs:

    - :class:`geopandas.GeoSeries`
    - :class:`geopandas.GeoDataFrame`
    """
    if isinstance(gpdf, gpGeoSeries):
        return GeoSeries(gpdf)
    elif isinstance(gpdf, gpGeoDataFrame):
        return GeoDataFrame(gpdf)
    elif isinstance(gpdf, pd.Series):
        raise TypeError("Mixed pandas/geometry types not supported yet.")
    else:
        raise TypeError
