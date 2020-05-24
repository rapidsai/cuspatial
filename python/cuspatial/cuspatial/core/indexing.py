# Copyright (c) 2020, NVIDIA CORPORATION.

import warnings

from cudf.core import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.quadtree import (
    quadtree_on_points as cpp_quadtree_on_points,
)
from cuspatial.utils.column_utils import normalize_point_columns


def quadtree_on_points(
    xs, ys, x_min, x_max, y_min, y_max, scale, max_depth, min_size
):
    """ Construct a quadtree from a set of points for a given area-of-interest
        bounding box.

    Parameters
    ----------
    {params}
    quadtree  : DataFrame of key, level, is_node, length, and offset columns
    """

    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    x_min, x_max, y_min, y_max = (
        min(x_min, x_max),
        max(x_min, x_max),
        min(y_min, y_max),
        max(y_min, y_max),
    )

    min_scale = max(x_max - x_min, y_max - y_min) / ((1 << max_depth) + 2)
    if scale < min_scale:
        warnings.warn(
            "scale {} is less than required minimum ".format(scale)
            + "scale {}. Clamping to minimum scale".format(min_scale)
        )

    points_order, quadtree = cpp_quadtree_on_points(
        xs,
        ys,
        x_min,
        x_max,
        y_min,
        y_max,
        max(scale, min_scale),
        max_depth,
        min_size,
    )
    return Series(points_order), DataFrame._from_table(quadtree)
