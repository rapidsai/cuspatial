# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf

from geopandas.geoseries import GeoSeries as gpGeoSeries
from geopandas.geoseries import is_geometry_type
from geopandas import GeoDataFrame as gpGeoDataFrame

from cuspatial.geometry.geoseries import GeoSeries
from cuspatial.geometry.geodataframe import GeoDataFrame


def from_geoseries(geoseries):
    cugs = GeoSeries(geoseries)
    cugs.index = cudf.from_pandas(geoseries.index)
    return cugs


def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.

    Possible inputs:
    GeoSeries
    GeoDataFrame
    """
    if isinstance(gpdf, gpGeoSeries):
        return from_geoseries(gpdf)
    if isinstance(gpdf, gpGeoDataFrame):
        gdf = GeoDataFrame()
        for col in gpdf.columns:
            if is_geometry_type(gpdf[col]):
                gdf[col] = from_geoseries(gpdf[col])
            else:
                gdf[col] = cudf.from_pandas(gpdf[col])
        gdf.index = gpdf.index
        return gdf
