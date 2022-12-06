# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import Series

import cuspatial


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
    result = Series([False] * len(test_points_x))
    result[point_indices[polygons_and_points["point_index"]]] = True
    return result
