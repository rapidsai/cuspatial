# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf.core import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.quadtree import (
    quadtree_on_points as cpp_quadtree_on_points,
)
from cuspatial.utils.column_utils import normalize_point_columns


def quadtree_on_points(xs, ys, x1, y1, x2, y2, scale, num_levels, min_size):
    """ Construct a quadtree from a set of points for a given area-of-interest
        bounding box.

    Parameters
    ----------
    {params}
    quadtree  : DataFrame of key, level, is_node, length, and offset columns
    """

    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))

    points_order, quadtree = cpp_quadtree_on_points(
        xs,
        ys,
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
        scale,
        num_levels,
        min_size,
    )
    return Series(points_order), DataFrame._from_table(quadtree)
