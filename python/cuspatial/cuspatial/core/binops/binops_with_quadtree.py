# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import Series
from cudf.core.column import as_column

import cuspatial
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils.column_utils import normalize_point_columns


def contains(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """Compute from a set of points and a set of polygons which points fall
    within each polygon. Note that `polygons_(x,y)` must be specified as
    closed polygons: the first and last coordinate of each polygon must be
    the same.

    Parameters
    ----------
    test_points_x
        x-coordinate of test points
    test_points_y
        y-coordinate of test points
    poly_offsets
        beginning index of the first ring in each polygon
    poly_ring_offsets
        beginning index of the first point in each ring
    poly_points_x
        x closed-coordinate of polygon points
    poly_points_y
        y closed-coordinate of polygon points

    Examples
    --------

    Test whether 3 points fall within either of two polygons

    # TODO: Examples

    note
    input Series x and y will not be index aligned, but computed as
    sequential arrays.

    note
    poly_ring_offsets must contain only the rings that make up the polygons
    indexed by poly_offsets. If there are rings in poly_ring_offsets that
    are not part of the polygons in poly_offsets, results are likely to be
    incorrect and behavior is undefined.

    Returns
    -------
    result : cudf.DataFrame
        A DataFrame of boolean values indicating whether each point falls
        within each polygon.
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

    QUADTREE = False
    if QUADTREE:
        scale = 5
        max_depth = 7
        min_size = 125
        x_max = poly_points_x.max()
        x_min = poly_points_x.min()
        y_max = poly_points_y.max()
        y_min = poly_points_y.min()
        point_indices, quadtree = cuspatial.quadtree_on_points(
            test_points_x,
            test_points_y,
            x_min,
            x_max,
            y_min,
            y_max,
            scale,
            max_depth,
            min_size,
        )
        poly_bboxes = cuspatial.polygon_bounding_boxes(
            poly_offsets, poly_ring_offsets, poly_points_x, poly_points_y
        )
        intersections = cuspatial.join_quadtree_and_bounding_boxes(
            quadtree, poly_bboxes, x_min, x_max, y_min, y_max, scale, max_depth
        )
        polygons_and_points = cuspatial.quadtree_point_in_polygon(
            intersections,
            quadtree,
            point_indices,
            test_points_x,
            test_points_y,
            poly_offsets,
            poly_ring_offsets,
            poly_points_x,
            poly_points_y,
        )
        return polygons_and_points
    else:
        pip_result = cpp_point_in_polygon(
            test_points_x,
            test_points_y,
            as_column(poly_offsets, dtype="int32"),
            as_column(poly_ring_offsets, dtype="int32"),
            poly_points_x,
            poly_points_y,
        )
        return Series(pip_result, dtype="bool")
