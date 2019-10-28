# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame, Series

from cuspatial._lib.shapefile_reader import (
    cpp_read_polygon_shapefile
)

def read_polygon_shapefile(filename):
    """Reads a pair of .shp and .shx files into a cudf DataFrame"""
    result = cpp_read_polygon_shapefile(filename)
    # print(Series(result[0]).shape)
    # print(Series(result[1]).shape)
    # print(Series(result[2]).shape)
    # print(Series(result[3]).shape)
    return DataFrame(
        {
            "f_pos": result[0],
            "r_pos": result[1],
        }), DataFrame(
        {
            "x": result[2],
            "y": result[3],
        })
