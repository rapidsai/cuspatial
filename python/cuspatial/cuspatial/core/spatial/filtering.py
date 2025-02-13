# Copyright (c) 2022-2025, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import ColumnBase, as_column

from cuspatial._lib import points_in_range
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import contains_only_points


def points_in_spatial_window(points: GeoSeries, min_x, max_x, min_y, max_y):
    """Return only the subset of coordinates that fall within a
    rectangular window.

    A point `(x, y)` is inside the query window if and only if
    ``min_x < x < max_x AND min_y < y < max_y``

    The window is specified by minimum and maximum x and y
    coordinates.

    Parameters
    ----------
    points: GeoSeries
        A geoseries of points
    min_x: float
        lower x-coordinate of the query window
    max_x: float
        upper x-coordinate of the query window
    min_y: float
        lower y-coordinate of the query window
    max_y: float
        upper y-coordinate of the query window

    Returns
    -------
    result : GeoSeries
        subset of `points` above that fall within the window

    Notes
    -----
    * Swaps ``min_x`` and ``max_x`` if ``min_x > max_x``
    * Swaps ``min_y`` and ``max_y`` if ``min_y > max_y``
    """

    if len(points) == 0:
        return GeoSeries([])

    if not contains_only_points(points):
        raise ValueError("GeoSeries must contain only points.")

    xs = as_column(points.points.x).to_pylibcudf(mode="read")
    ys = as_column(points.points.y).to_pylibcudf(mode="read")

    plc_table = points_in_range.points_in_range(
        min_x, max_x, min_y, max_y, xs, ys
    )
    res_xy = DataFrame._from_data(
        {
            name: ColumnBase.from_pylibcudf(col)
            for name, col in zip(["x", "y"], plc_table.columns())
        }
    ).interleave_columns()
    return GeoSeries.from_points_xy(res_xy)
