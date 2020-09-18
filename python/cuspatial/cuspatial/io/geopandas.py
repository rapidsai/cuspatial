# Copyright (c) 2020, NVIDIA CORPORATION.

from collections.abc import Iterable
from geopandas.geoseries import GeoSeries as gpGeoSeries
from numba import cuda

import cupy as cp
import numpy as np

import cudf

from cuspatial import GeoSeries


def cpu_pack_geometries(i, arrays, offsets, output):
    print(i, arrays, offsets, output)
    if i < offsets.size:
        start = offsets[i-1] if i != 0 else 0
        end = offsets[i]
        arr = arrays[i]
        for j in range(start, end):
            print(j)
            value = arr[j-start]
            output[j] = value



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
    if isinstance(gpdf, Point):
        print('It is a point')
    print(gpdf)

