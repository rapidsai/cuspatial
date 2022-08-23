# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib import spatial_window
from cuspatial.utils.column_utils import normalize_point_columns


def points_in_spatial_window(min_x, max_x, min_y, max_y, xs, ys):
    """Return only the subset of coordinates that fall within a
    rectangular window.

    A point `(x, y)` is inside the query window if and only if
    ``min_x < x < max_x AND min_y < y < max_y``

    The window is specified by minimum and maximum x and y
    coordinates.

    Parameters
    ----------
    min_x
        lower x-coordinate of the query window
    max_x
        upper x-coordinate of the query window
    min_y
        lower y-coordinate of the query window
    max_y
        upper y-coordinate of the query window
    xs
        column of x-coordinates that may fall within the window
    ys
        column of y-coordinates that may fall within the window

    Returns
    -------
    result : cudf.DataFrame
        subset of `(x, y)` pairs above that fall within the window

    Notes
    -----
    * Swaps ``min_x`` and ``max_x`` if ``min_x > max_x``
    * Swaps ``min_y`` and ``max_y`` if ``min_y > max_y``
    """
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *spatial_window.points_in_spatial_window(
            min_x, max_x, min_y, max_y, xs, ys
        )
    )
