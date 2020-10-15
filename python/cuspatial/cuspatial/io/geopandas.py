# Copyright (c) 2020, NVIDIA CORPORATION.

import cudf

from geopandas.geoseries import GeoSeries as gpGeoSeries
from geopandas import GeoDataFrame as gpGeoDataFrame

from cuspatial.geometry.geoseries import GeoSeries


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
    """
    if isinstance(gpdf, gpGeoSeries):
        return from_geoseries(gpdf)
    if isinstance(gpdf, gpGeoDataFrame):
        geo_columns = gpdf.columns[gpdf.dtypes == "geometry"]
        non_geo_columns = gpdf[gpdf.columns[gpdf.dtypes != "geometry"]]
        gdf = cudf.from_pandas(non_geo_columns)
        for col in geo_columns:
            cu_series = from_geoseries(gpdf[col])
            gdf[col] = from_geoseries(gpdf[col])
        gdf.index = gpdf.index
        return gdf
