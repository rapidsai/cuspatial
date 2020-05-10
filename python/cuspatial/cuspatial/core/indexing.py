# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf.core import DataFrame
from cudf.core.column import as_column

from cuspatial._lib.quadtree import (
    quadtree_on_points as cpp_quadtree_on_points
)

import numpy as np

def quadtree_on_points(xs, ys, x1, y1, x2, y2, scale, num_levels, min_size):
    """ Construct a quadtree from a set of points for a given area-of-interest
        bounding box.

    Parameters
    ----------
    {params}
    quadtree  : DataFrame of key, level, is_node, length, and offset columns
    """

    # Todo: Replace with call to `normalize_point_columns` in trajectory PR
    xs = as_column(xs)
    ys = as_column(ys)
    dtype = np.result_type(xs.dtype, ys.dtype)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32 if dtype.itemsize <= 4 else np.float64
    xs, ys = xs.astype(dtype), ys.astype(dtype)

    return DataFrame._from_table(cpp_quadtree_on_points(
        xs, ys,
        min(x1, x2), min(y1, y2),
        max(x1, x2), max(y1, y2),
        scale, num_levels, min_size
    ))
