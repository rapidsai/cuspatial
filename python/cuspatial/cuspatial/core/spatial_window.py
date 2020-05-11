# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib.spatial_window import points_in_spatial_window


def window_points(left, bottom, right, top, xs, ys):
    """ Return only the subset of coordinates that fall within the numerically
    closed borders [,] of the defined bounding box.

    params
    left: x coordinate of window left boundary
    bottom: y coordinate of window bottom boundary
    right: x coordinate of window right boundary
    top: y coordinate of window top boundary
    xs: Series of x coordinates that may fall within the window
    ys: Series of y coordinates that may fall within the window

    Parameters
    ----------
    {params}

    Returns
    -------
    DataFrame: subset of x, y pairs above that fall within the window
    """
    result = points_in_spatial_window(left, right, bottom, top, as_column(xs),
                                      as_column(ys))
    return DataFrame._from_table(result)
