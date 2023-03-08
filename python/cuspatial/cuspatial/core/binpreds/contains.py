# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from math import ceil, sqrt

from cudf import DataFrame, Series
from cudf.core.column import NumericalColumn, as_column

import cuspatial
from cuspatial._lib.pairwise_point_in_polygon import (
    pairwise_point_in_polygon as cpp_pairwise_point_in_polygon,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils.column_utils import normalize_point_columns


def contains_properly_quadtree(
    test_points,
    polygons,
):
    """Compute from a series of points and a series of polygons which points
    are properly contained within the corresponding polygon. Polygon A contains
    Point B properly if B intersects the interior of A but not the boundary (or
    exterior).

    Note that polygons must be closed: the first and last vertex of each
    polygon must be the same.

    Parameters
    ----------
    test_points
        GeoSeries of points to test for containment
    polygons
        GeoSeries of polygons to test for containment

    Returns
    -------
    result : cudf.Series
        A Series of boolean values indicating whether each point falls
        within its corresponding polygon.
    """

    scale = -1
    max_depth = 15
    min_size = ceil(sqrt(len(test_points)))
    x_max = polygons.polygons.x.max()
    x_min = polygons.polygons.x.min()
    y_max = polygons.polygons.y.max()
    y_min = polygons.polygons.y.min()
    point_indices, quadtree = cuspatial.quadtree_on_points(
        test_points,
        x_min,
        x_max,
        y_min,
        y_max,
        scale,
        max_depth,
        min_size,
    )
    poly_bboxes = cuspatial.polygon_bounding_boxes(polygons)
    intersections = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree, poly_bboxes, x_min, x_max, y_min, y_max, scale, max_depth
    )
    polygons_and_points = cuspatial.quadtree_point_in_polygon(
        intersections,
        quadtree,
        point_indices,
        test_points,
        polygons,
    )
    polygons_and_points["point_index"] = point_indices.iloc[
        polygons_and_points["point_index"]
    ].reset_index(drop=True)
    polygons_and_points["part_index"] = polygons_and_points["polygon_index"]
    polygons_and_points.drop("polygon_index", axis=1, inplace=True)
    return polygons_and_points


def contains_properly_pairwise(
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
        x-coordinates of points to test for containment
    test_points_y
        y-coordinates of points to test for containment
    poly_offsets
        offsets of the first ring in each polygon
    poly_ring_offsets
        offsets of the first point in each ring
    poly_points_x
        x-coordinates of polygon points
    poly_points_y
        y-coordinates of polygon points

    Returns
    -------
    result : cudf.DataFrame
        A DataFrame of boolean values indicating whether each point falls
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
    poly_offsets = poly_offsets[:-1]
    poly_ring_offsets = poly_ring_offsets[:-1]
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
    if isinstance(pip_result, NumericalColumn):
        result = DataFrame(pip_result)
    else:
        result = DataFrame(
            {k: v for k, v in enumerate(pip_result)}, dtype="bool"
        )
    return result
