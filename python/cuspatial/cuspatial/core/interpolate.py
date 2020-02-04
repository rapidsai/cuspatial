# Copyright (c) 2019, NVIDIA CORPORATION.

from cuspatial._lib.interpolate import (
    cubicspline_column
)

from cudf import DataFrame


def cubic_spline(x, y, ids_and_end_coordinates):
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
    ids_c = ids_and_end_coordinates._column
    result = cubicspline_column(x_c, y_c, ids_c)
    return DataFrame(result._columns)

def cubic_spline_2(x, y, ids_and_prefix_sum):
    result = cubicspline_full(x, y, ids_and_prefix_sum())
    return result
