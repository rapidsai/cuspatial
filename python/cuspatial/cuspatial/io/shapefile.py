# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame, Series

from cuspatial._lib.shapefile_reader import cpp_read_polygon_shapefile


def read_polygon_shapefile(filename):
    """Reads a shapefile into GPU memory."""
    result = cpp_read_polygon_shapefile(filename)
    f_pos = Series(result[0], name="f_pos")
    r_pos = Series(result[1], name="r_pos")
    return (f_pos, r_pos, DataFrame({"x": result[2], "y": result[3]}))
