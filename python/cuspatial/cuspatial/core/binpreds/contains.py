# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import Series
from cudf.core.column import as_column

from cuspatial._lib.pairwise_point_in_polygon import (
    pairwise_point_in_polygon as cpp_pairwise_point_in_polygon,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils.column_utils import normalize_point_columns


def contains_properly(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """Compute from a series of points and a series of polygons which points
    are properly contained within the corresponding polygon. Polygon A contains
    Point B properly if B intersects the interior of A but not the boundary (or
    exterior).

    Note that polygons must be closed: the first and last vertex of each
    polygon must be the same.

    Parameters
    ----------
    test_points_x
        x-coordinate of points to test for containment
    test_points_y
        y-coordinate of points to test for containment
    poly_offsets
        beginning index of the first ring in each polygon
    poly_ring_offsets
        beginning index of the first point in each ring
    poly_points_x
        x-coordinates of polygon vertices
    poly_points_y
        y-coordinates of polygon vertices

    Returns
    -------
    result : cudf.Series
        A Series of boolean values indicating whether each point falls
        within its corresponding polygon.
    """

    if len(poly_offsets) == 0:
        return Series()
    (
        test_points_x,
        test_points_y,
        poly_points_x,
        poly_points_y,
    ) = normalize_point_columns(
        as_column(test_points_x),
        as_column(test_points_y),
        as_column(poly_points_x),
        as_column(poly_points_y),
    )
    poly_offsets_column = as_column(poly_offsets, dtype="int32")
    poly_ring_offsets_column = as_column(poly_ring_offsets, dtype="int32")

    if len(test_points_x) == len(poly_offsets):
        pip_result = cpp_pairwise_point_in_polygon(
            test_points_x,
            test_points_y,
            poly_offsets_column,
            poly_ring_offsets_column,
            poly_points_x,
            poly_points_y,
        )
    else:
        pip_result = cpp_point_in_polygon(
            test_points_x,
            test_points_y,
            poly_offsets_column,
            poly_ring_offsets_column,
            poly_points_x,
            poly_points_y,
        )

    result = Series(pip_result, dtype="bool")
    return result
