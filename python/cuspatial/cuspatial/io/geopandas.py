# Copyright (c) 2020, NVIDIA CORPORATION.

from collections.abc import Iterable
from geopandas.geoseries import GeoSeries
from numba import cuda
from shapely.geometry import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
)

import cupy as cp
import numpy as np

import cudf


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

def _cpu_pack_point(i, point, offset, output):
    output[0+offset] = point[0]
    output[1+offset] = point[1]


def from_geoseries(geoseries):
    # do in parallel: compute offsets of the geoseries
    # do in parallel: accumulate the size of the geoseries
    # allocate memory for the full size
    # do in parallel: copy each geoseries into the proper offset
    # return cuSeries
    # offsets?
    offsets = []
    for geometry in geoseries:
        if isinstance(geometry, Point):
            offsets.append(len(geometry.xy))
        elif isinstance(geometry, MultiPoint):
            offsets.append(len(geometry))
        elif isinstance(geometry, LineString):
            offsets.append(len(geometry.xy))
        elif isinstance(geometry, MultiLineString):
            offsets.append(len(geometry.xy))
    print(offsets)

    current_offset = 0

    for geometry in geoseries:
        if isinstance(geometry, Point):
            arr = np.array(geometry)
            offset = len(arr)
            cpu_buffer = np.zeros(2)
            _cpu_pack_point(0, arr, current_offset, cpu_buffer)
            current_offset = current_offset + offset
        elif isinstance(geometry, Iterable):
            cp_arrays = np.array(list(map(lambda x: np.array(x), geometry)))
            lengths = cudf.Series(map(lambda x: len(x), cp_arrays))
            offsets = lengths.cumsum()
            size = lengths.sum()
            cpu_buffer = np.zeros(size)
            if len(cp_arrays) > 1:
                for i in range(len(cp_arrays)):
                    cpu_pack_geometries(i, cp_arrays, offsets, cpu_buffer)

    return (cudf.Series(cpu_buffer), offsets)
    

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
    if isinstance(gpdf, GeoSeries):
        return from_geoseries(gpdf)
    if isinstance(gpdf, Point):
        print('It is a point')
    print(gpdf)

