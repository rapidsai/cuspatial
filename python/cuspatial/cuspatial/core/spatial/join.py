# Copyright (c) 2022, NVIDIA CORPORATION.

import warnings

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib import spatial_join
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils import gis_utils
from cuspatial.utils.column_utils import normalize_point_columns


def point_in_polygon(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """Compute from a set of points and a set of polygons which points fall
    within which polygons. Note that `polygons_(x,y)` must be specified as
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

    >>> result = cuspatial.point_in_polygon(
        [0, -8, 6.0],                             # test_points_x
        [0, -8, 6.0],                             # test_points_y
        cudf.Series([0, 1], index=['nyc', 'hudson river']), # poly_offsets
        [0, 3],                                   # ring_offsets
        [-10, 5, 5, -10, 0, 10, 10, 0],           # poly_points_x
        [-10, -10, 5, 5, 0, 0, 10, 10],           # poly_points_y
    )
    # The result of point_in_polygon is a DataFrame of Boolean
    # values indicating whether each point (rows) falls within
    # each polygon (columns).
    >>> print(result)
                nyc            hudson river
    0          True          True
    1          True         False
    2         False          True
    # Point 0: (0, 0) falls in both polygons
    # Point 1: (-8, -8) falls in the first polygon
    # Point 2: (6.0, 6.0) falls in the second polygon

    Notes
    -----

    * input Series x and y will not be index aligned, but computed as
    sequential arrays.

    * poly_ring_offsets must contain only the rings that make up the polygons
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
        return DataFrame()

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

    result = cpp_point_in_polygon(
        test_points_x,
        test_points_y,
        as_column(poly_offsets, dtype="int32"),
        as_column(poly_ring_offsets, dtype="int32"),
        poly_points_x,
        poly_points_y,
    )

    result = gis_utils.pip_bitmap_column_to_binary_array(
        polygon_bitmap_column=result, width=len(poly_offsets)
    )
    result = DataFrame(result)
    result = DataFrame._from_data(
        {name: col.astype("bool") for name, col in result._data.items()}
    )
    result.columns = [x for x in list(reversed(poly_offsets.index))]
    result = result[list(reversed(result.columns))]
    return result


def join_quadtree_and_bounding_boxes(
    quadtree, bounding_boxes, x_min, x_max, y_min, y_max, scale, max_depth
):
    """Search a quadtree for polygon or linestring bounding box intersections.

    Parameters
    ----------
    quadtree : cudf.DataFrame
        A complete quadtree for a given area-of-interest bounding box.
    bounding_boxes : cudf.DataFrame
        Minimum bounding boxes for a set of polygons or linestrings
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

        bbox_offset : cudf.Series
            Indices for each bbox that intersects with the quadtree.
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

    return DataFrame._from_data(
        *spatial_join.join_quadtree_and_bounding_boxes(
            quadtree,
            bounding_boxes,
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
    """Test whether the specified points are inside any of the specified
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

        polygon_index : cudf.Series
            Index of containing polygon.
        point_index : cudf.Series
            Index of contained point. This index refers to ``point_indices``,
            so it is an index to an index.
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
    return DataFrame._from_data(
        *spatial_join.quadtree_point_in_polygon(
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


def quadtree_point_to_nearest_linestring(
    linestring_quad_pairs,
    quadtree,
    point_indices,
    points_x,
    points_y,
    linestring_offsets,
    linestring_points_x,
    linestring_points_y,
):
    """Finds the nearest linestring to each point in a quadrant, and computes
    the distances between each point and linestring.

    Uses the table of (linestring, quadrant) pairs returned by
    ``cuspatial.join_quadtree_and_bounding_boxes`` to ensure distances are
    computed only for the points in the same quadrant as each linestring.

    Parameters
    ----------
    linestring_quad_pairs: cudf.DataFrame
        Table of (linestring, quadrant) index pairs returned by
        ``cuspatial.join_quadtree_and_bounding_boxes``.
    quadtree : cudf.DataFrame
        A complete quadtree for a given area-of-interest bounding box.
    point_indices : cudf.Series
        Sorted point indices returned by ``cuspatial.quadtree_on_points``
    points_x : cudf.Series
        x-coordinates of points used to construct the quadtree.
    points_y : cudf.Series
        y-coordinates of points used to construct the quadtree.
    linestring_offsets : cudf.Series
        Begin index of the first point in each linestring.
    poly_points_x : cudf.Series
        Linestring point x-coordinates.
    poly_points_y : cudf.Series
        Linestring point y-coordinates.

    Returns
    -------
    result : cudf.DataFrame
        Indices for each point and its nearest linestring, and the distance
        between the two.

        point_index : cudf.Series
            Index of point. This index refers to ``point_indices``, so it is
            an index to an index.
        linestring_index : cudf.Series
            Index of the nearest linestring to the point.
        distance : cudf.Series
            Distance between point and its nearest linestring.
    """
    (
        points_x,
        points_y,
        linestring_points_x,
        linestring_points_y,
    ) = normalize_point_columns(
        as_column(points_x),
        as_column(points_y),
        as_column(linestring_points_x),
        as_column(linestring_points_y),
    )
    return DataFrame._from_data(
        *spatial_join.quadtree_point_to_nearest_linestring(
            linestring_quad_pairs,
            quadtree,
            as_column(point_indices, dtype="uint32"),
            points_x,
            points_y,
            as_column(linestring_offsets, dtype="uint32"),
            linestring_points_x,
            linestring_points_y,
        )
    )
