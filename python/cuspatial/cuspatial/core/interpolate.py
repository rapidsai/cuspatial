# Copyright (c) 2020, NVIDIA CORPORATION.

from cuspatial._lib.interpolate import (
    cubicspline_full
)

from cudf import DataFrame, Series
from cudf.core.index import RangeIndex

import numpy as np


def cubic_spline_2(x, y, ids, prefix_sums):
    """
    Fits each column of the input DataFrame `y` to a hermetic cubic spline.

    Parameters
    ----------
    x : cudf.Series
        time sample values. Must be monotonically increasing.
    y : cudf.DataFrame
        columns to have curves fit to according to x
    ids_and_end_coordinates: cudf.DataFrame
                             ids and final positions of each set of
                             trajectories

    Returns
    -------
    m x n DataFrame of trajectory curve coefficients.
    m is len(ids_and_end_coordinates), n is 4 * len(y.columns)
    """
    x_c = x._column
    y_c = y._column
    ids_c = ids._column
    prefix_c = prefix_sums._column
    result_table = cubicspline_full(x_c, y_c, ids_c, prefix_c)
    result_table._index = RangeIndex(result_table._num_rows)
    result = DataFrame._from_table(result_table)
    return result


def cubic_spline_fit(c, points):
    return {}


class CubicSpline:
    def __init__(self, t, y, ids, size):
        # error protections:
        if not isinstance(t, Series):
            raise TypeError(
                "Error: input independent vars must be cudf Series"
            )
        if not isinstance(y, (Series, DataFrame)):
            raise TypeError(
                "Error: input dependent vars must be cudf Series or DataFrame"
            )
        if not len(t) == len(y):
            raise TypeError(
                "Error: dependent and independent vars have different length"
            )
        if not len(t) == len(ids):
            raise ValueError("error: id length doesn't match input length")
        if not (len(t) % size == 0):
            raise ValueError(
                "Error: length of input is not a multiple of size"
            )
        self.t = t
        self.y = y
        self.ids = ids
        self.size = size
        self._c = {}
        prefix = Series(np.arange(len(t) / size) * size)
        if isinstance(y, Series):
            self.c["y"] = cubic_spline_2(t, y, ids, prefix)
        else:
            for col in y.columns:
                self.c[col] = cubic_spline_2(t, y, ids, prefix)

    def __call__(self, coordinates):
        if isinstance(self.y, Series):
            return cubic_spline_fit(self.c, coordinates)
        else:
            result = DataFrame()
            for col in self.y.columns:
                result[col] = cubic_spline_fit(self.c[col], coordinates)
            return result
