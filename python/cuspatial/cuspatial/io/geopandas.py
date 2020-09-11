# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf import DataFrame, Series

def from_geopandas(gpdf):
    """
    Converts a geopandas mixed geometry dataframe into a cuspatial geometry
    dataframe.
    """
    print(gpdf)
    return DataFrame({})
