# Copyright (c) 2019, NVIDIA CORPORATION.

from cuspatial._lib.interpolate import (
    cpp_cubicspline
)

from cudf._libxx.table import _Table
from cudf import DataFrame


def cubic_spline(x, y, ids_and_end_coordinates):
    x_c = x._column
    y_c = _Table(y._columns)
    ids_c = _Table(ids_and_end_coordinates._columns)
    result = cpp_cubicspline(x_c, y_c, ids_c)
    return DataFrame(result.columns)
