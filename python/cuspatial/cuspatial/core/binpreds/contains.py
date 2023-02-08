# Copyright (c) 2022, NVIDIA CORPORATION.

from math import ceil, sqrt

from cudf import Series
from cudf.core.column import as_column

import cuspatial
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

    scale = (
        (test_points_x.std() + test_points_y.std()) / 2.0
        if len(test_points_x) > 1
        else 1.0
    )
    max_depth = 15
    min_size = ceil(sqrt(len(test_points_x)))
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
    polygons_and_points["point_index"] = point_indices.iloc[
        polygons_and_points["point_index"]
    ].reset_index(drop=True)
    polygons_and_points["part_index"] = polygons_and_points["polygon_index"]
    polygons_and_points.drop("polygon_index", axis=1, inplace=True)
    return polygons_and_points
