# Copyright (c) 2022, NVIDIA CORPORATION.

import warnings

from cudf import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.quadtree import (
    quadtree_on_points as cpp_quadtree_on_points,
)
from cuspatial.utils.column_utils import normalize_point_columns


def quadtree_on_points(
    xs, ys, x_min, x_max, y_min, y_max, scale, max_depth, max_size
):
    """Construct a quadtree from a set of points for a given area-of-interest
    bounding box.

    Parameters
    ----------
    xs
        Column of x-coordinates for each point.
    ys
        Column of y-coordinates for each point.
    x_min
        The lower-left x-coordinate of the area of interest bounding box.
    x_max
        The upper-right x-coordinate of the area of interest bounding box.
    y_min
        The lower-left y-coordinate of the area of interest bounding box.
    y_max
        The upper-right y-coordinate of the area of interest bounding box.
    scale
        Scale to apply to each point's distance from ``(x_min, y_min)``.
    max_depth
        Maximum quadtree depth.
    max_size
        Maximum number of points allowed in a node before it's split into
        4 leaf nodes.

    Returns
    -------
    result : tuple (cudf.Series, cudf.DataFrame)
        keys_to_points  : cudf.Series(dtype=np.int32)
            A column of sorted keys to original point indices
        quadtree        : cudf.DataFrame
            A complete quadtree for the set of input points

            key         : cudf.Series(dtype=np.int32)
                An int32 column of quadrant keys
            level       : cudf.Series(dtype=np.int8)
                An int8 column of quadtree levels
            is_internal_node : cudf.Series(dtype=np.bool_)
                A boolean column indicating whether the node is a quad or leaf
            length      : cudf.Series(dtype=np.int32)
                If this is a non-leaf quadrant (i.e. ``is_internal_node`` is
                ``True``), this column's value is the number of children in
                the non-leaf quadrant.

                Otherwise this column's value is the number of points
                contained in the leaf quadrant.
            offset      : cudf.Series(dtype=np.int32)
                If this is a non-leaf quadrant (i.e. ``is_internal_node`` is
                ``True``), this column's value is the position of the non-leaf
                quadrant's first child.

                Otherwise this column's value is the position of the leaf
                quadrant's first point.

    Notes
    -----

    * Swaps ``min_x`` and ``max_x`` if ``min_x > max_x``

    * Swaps ``min_y`` and ``max_y`` if ``min_y > max_y``

    * 2D coordinates are converted into a 1D Morton code by dividing each x/y
    by the ``scale``: (``(x - min_x) / scale`` and ``(y - min_y) / scale``).

    * `max_depth` should be less than 16, since Morton codes are represented
    as `uint32_t`. The eventual number of levels may be less than `max_depth`
    if the number of points is small or `max_size` is large.

    * All intermediate quadtree nodes will have fewer than `max_size` number of
    points. Leaf nodes are permitted (but not guaranteed) to have >= `max_size`
    number of points.

    Examples
    --------

    An example of selecting the ``max_size`` and ``scale`` based on input::

        >>> np.random.seed(0)
        >>> points = cudf.DataFrame({
                "x": cudf.Series(np.random.normal(size=120)) * 500,
                "y": cudf.Series(np.random.normal(size=120)) * 500,
            })

        >>> max_depth = 3
        >>> max_size = 50
        >>> min_x, min_y, max_x, max_y = (points["x"].min(),
                                          points["y"].min(),
                                          points["x"].max(),
                                          points["y"].max())
        >>> scale = max(max_x - min_x, max_y - min_y) // (1 << max_depth)
        >>> print(
                "max_size:   " + str(max_size) + "\\n"
                "num_points: " + str(len(points)) + "\\n"
                "min_x:      " + str(min_x) + "\\n"
                "max_x:      " + str(max_x) + "\\n"
                "min_y:      " + str(min_y) + "\\n"
                "max_y:      " + str(max_y) + "\\n"
                "scale:      " + str(scale) + "\\n"
            )
        max_size:   50
        num_points: 120
        min_x:      -1577.4949079170394
        max_x:      1435.877311993804
        min_y:      -1412.7015761122134
        max_y:      1492.572387431971
        scale:      301.0

        >>> key_to_point, quadtree = cuspatial.quadtree_on_points(
                points["x"],
                points["y"],
                min_x,
                max_x,
                min_y,
                max_y,
                scale, max_depth, max_size
            )

        >>> print(quadtree)
            key  level  is_internal_node  length  offset
        0     0      0             False      15       0
        1     1      0             False      27      15
        2     2      0             False      12      42
        3     3      0              True       4       8
        4     4      0             False       5     106
        5     6      0             False       6     111
        6     9      0             False       2     117
        7    12      0             False       1     119
        8    12      1             False      22      54
        9    13      1             False      18      76
        10   14      1             False       9      94
        11   15      1             False       3     103

        >>> print(key_to_point)
        0       63
        1       20
        2       33
        3       66
        4       19
            ...
        115    113
        116      3
        117     78
        118     98
        119     24
        Length: 120, dtype: int32
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

    key_to_point, quadtree = cpp_quadtree_on_points(
        xs,
        ys,
        x_min,
        x_max,
        y_min,
        y_max,
        max(scale, min_scale),
        max_depth,
        max_size,
    )
    return Series(key_to_point), DataFrame._from_data(*quadtree)
