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

def _cpu_pack_point(point, offset, output):
    output[0+offset] = point[0]
    output[1+offset] = point[1]


def _cpu_pack_multipoint(multipoint, offset, output):
    multipoint_array = np.array(list(map(lambda x: np.array(x), multipoint)))
    for point in multipoint_array:
        _cpu_pack_point(point, offset, output)
        offset = offset + 2


def _cpu_pack_linestring(linestring, offset, output):
    linestring_array = np.array(
        list(map(lambda x: np.array(x), linestring.coords))
    )
    for point in linestring_array:
        _cpu_pack_point(point, offset, output)
        offset = offset + 2


def _cpu_pack_multimultilinestring(multilinestring, offset, output):
    multilinestring_array = np.array(
        list(map(lambda x: np.array(x), multilinestring.coords))
    )
    for point in multilinestring_array:
        _cpu_pack_point(point, offset, output)
        offset = offset + 2


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
            offsets.append((1 + np.arange(len(geometry))) * 2)
        elif isinstance(geometry, LineString):
            offsets.append((1 + np.arange(len(geometry.xy))) * 2)
        elif isinstance(geometry, MultiLineString):
            breakpoint()
            offsets.append(len(geometry.coords))
    offsets = np.array(offsets)
    print(offsets)

    current_offset = 0
    cpu_buffer = np.zeros(offsets.max())

    for geometry in geoseries:
        if isinstance(geometry, Point):
            arr = np.array(geometry)
            offset = len(arr)
            _cpu_pack_point(arr, current_offset, cpu_buffer)
            current_offset = current_offset + offset
        elif isinstance(geometry, MultiPoint):
            _cpu_pack_multipoint(geometry, current_offset, cpu_buffer)
        elif isinstance(geometry, LineString):
            _cpu_pack_linestring(geometry, current_offset, cpu_buffer)

    return (cudf.Series(cpu_buffer), cudf.Series(offsets.flatten()))
    

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

