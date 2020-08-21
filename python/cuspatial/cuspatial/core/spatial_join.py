# Copyright (c) 2020, NVIDIA CORPORATION.

import warnings

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib import spatial_join

from cuspatial.utils.column_utils import normalize_point_columns


def join_quadtree_and_bounding_boxes(
    quadtree, poly_bounding_boxes, x_min, x_max, y_min, y_max, scale, max_depth
):
    """ Search a quadtree for polygon or polyline bounding box intersections.

    Parameters
    ----------
    quadtree : cudf.DataFrame
        A complete quadtree for a given area-of-interest bounding box.
    poly_bounding_boxes : cudf.DataFrame
        Minimum bounding boxes for a set of polygons or polylines
    x_min
        The lower-left x-coordinate of the area of interest bounding box
    x_max
        The upper-right x-coordinate of the area of interest bounding box
    min_y
        The lower-left y-coordinate of the area of interest bounding box
    max_y
        The upper-right y-coordinate of the area of interest bounding box
    scale
        Scale to apply to each point's distance from ``(x_min, y_min)``
    max_depth
        Maximum quadtree depth at which to stop testing for intersections

    Returns
    -------
    result : cudf.DataFrame
        Indices for each intersecting bounding box and leaf quadrant.

        poly_offset : cudf.Series
            Indices for each poly bbox that intersects with the quadtree.
        quad_offset : cudf.Series
            Indices for each leaf quadrant intersecting with a poly bbox.

    Notes
    -----
    * Swaps ``min_x`` and ``max_x`` if ``min_x > max_x``
    * Swaps ``min_y`` and ``max_y`` if ``min_y > max_y``
    """
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

    return DataFrame._from_table(
        spatial_join.join_quadtree_and_bounding_boxes(
            quadtree,
            poly_bounding_boxes,
            x_min,
            x_max,
            y_min,
            y_max,
            max(scale, min_scale),
            max_depth,
        )
    )


def quadtree_point_in_polygon(
    poly_quad_pairs,
    quadtree,
    point_indices,
    points_x,
    points_y,
    poly_offsets,
    ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """ Test whether the specified points are inside any of the specified
    polygons.

    Uses the table of (polygon, quadrant) pairs returned by
    ``cuspatial.join_quadtree_and_bounding_boxes`` to ensure only the points
    in the same quadrant as each polygon are tested for intersection.

    This pre-filtering can dramatically reduce number of points tested per
    polygon, enabling faster intersection-testing at the expense of extra
    memory allocated to store the quadtree and sorted point_indices.

    Parameters
    ----------
    poly_quad_pairs: cudf.DataFrame
        Table of (polygon, quadrant) index pairs returned by
        ``cuspatial.join_quadtree_and_bounding_boxes``.
    quadtree : cudf.DataFrame
        A complete quadtree for a given area-of-interest bounding box.
    point_indices : cudf.Series
        Sorted point indices returned by ``cuspatial.quadtree_on_points``
    points_x : cudf.Series
        x-coordinates of points used to construct the quadtree.
    points_y : cudf.Series
        y-coordinates of points used to construct the quadtree.
    poly_offsets : cudf.Series
        Begin index of the first ring in each polygon.
    ring_offsets : cudf.Series
        Begin index of the first point in each ring.
    poly_points_x : cudf.Series
        Polygon point x-coodinates.
    poly_points_y : cudf.Series
        Polygon point y-coodinates.

    Returns
    -------
    result : cudf.DataFrame
        Indices for each intersecting point and polygon pair.

        point_offset : cudf.Series
            Indices of each point that intersects with a polygon.
        polygon_offset : cudf.Series
            Indices of each polygon with which a point intersected.
    """

    (
        points_x,
        points_y,
        poly_points_x,
        poly_points_y,
    ) = normalize_point_columns(
        as_column(points_x),
        as_column(points_y),
        as_column(poly_points_x),
        as_column(poly_points_y),
    )
    return DataFrame._from_table(
        spatial_join.quadtree_point_in_polygon(
            poly_quad_pairs,
            quadtree,
            as_column(point_indices, dtype="uint32"),
            points_x,
            points_y,
            as_column(poly_offsets, dtype="uint32"),
            as_column(ring_offsets, dtype="uint32"),
            poly_points_x,
            poly_points_y,
        )
    )


def quadtree_point_to_nearest_polyline(
    poly_quad_pairs,
    quadtree,
    point_indices,
    points_x,
    points_y,
    poly_offsets,
    poly_points_x,
    poly_points_y,
):
    """ Finds the nearest polyline to each point in a quadrant, and computes
    the distances between each point and polyline.

    Uses the table of (polyline, quadrant) pairs returned by
    ``cuspatial.join_quadtree_and_bounding_boxes`` to ensure distances are
    computed only for the points in the same quadrant as each polyline.

    Parameters
    ----------
    poly_quad_pairs: cudf.DataFrame
        Table of (polyline, quadrant) index pairs returned by
        ``cuspatial.join_quadtree_and_bounding_boxes``.
    quadtree : cudf.DataFrame
        A complete quadtree for a given area-of-interest bounding box.
    point_indices : cudf.Series
        Sorted point indices returned by ``cuspatial.quadtree_on_points``
    points_x : cudf.Series
        x-coordinates of points used to construct the quadtree.
    points_y : cudf.Series
        y-coordinates of points used to construct the quadtree.
    poly_offsets : cudf.Series
        Begin index of the first point in each polyline.
    poly_points_x : cudf.Series
        Polyline point x-coodinates.
    poly_points_y : cudf.Series
        Polyline point y-coodinates.

    Returns
    -------
    result : cudf.DataFrame
        Indices for each point and its nearest polyline, and the distance
        between the two.

        point_offset : cudf.Series
            Indices of each point that intersects with a polyline.
        polyline_offset : cudf.Series
            Indices of each polyline with which a point intersected.
        distance : cudf.Series
            Distances between each point and its nearest polyline.
    """
    (
        points_x,
        points_y,
        poly_points_x,
        poly_points_y,
    ) = normalize_point_columns(
        as_column(points_x),
        as_column(points_y),
        as_column(poly_points_x),
        as_column(poly_points_y),
    )
    return DataFrame._from_table(
        spatial_join.quadtree_point_to_nearest_polyline(
            poly_quad_pairs,
            quadtree,
            as_column(point_indices, dtype="uint32"),
            points_x,
            points_y,
            as_column(poly_offsets, dtype="uint32"),
            poly_points_x,
            poly_points_y,
        )
    )
