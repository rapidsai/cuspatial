# Copyright (c) 2022-2023, NVIDIA CORPORATION.

import warnings

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial import GeoSeries
from cuspatial._lib import spatial_join
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_points,
    contains_only_polygons,
)
from cuspatial.utils.join_utils import pip_bitmap_column_to_binary_array


def point_in_polygon(points: GeoSeries, polygons: GeoSeries):
    """Compute from a set of points and a set of polygons which points fall
    within which polygons. Note that `polygons_(x,y)` must be specified as
    closed polygons: the first and last coordinate of each polygon must be
    the same.

    Parameters
    ----------
    points : GeoSeries
        A Series of points to test
    polygons: GeoSeries
        A Series of polygons to test

    Examples
    --------
    Test whether 3 points fall within either of two polygons

    >>> result = cuspatial.point_in_polygon(
        GeoSeries([Point(0, 0), Point(-8, -8), Point(6, 6)]),
        GeoSeries([
            Polygon([(-10, -10), (5, -10), (5, 5), (-10, 5), (-10, -10)]),
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        ], index=['nyc', 'hudson river'])
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

    Returns
    -------
    result : cudf.DataFrame
        A DataFrame of boolean values indicating whether each point falls
        within each polygon.
    """

    if len(polygons) == 0:
        return DataFrame()

    # The C++ API only supports single-polygon, reject if input has
    # multipolygons
    if len(polygons.polygons.part_offset) != len(
        polygons.polygons.geometry_offset
    ):
        raise ValueError("GeoSeries cannot contain multipolygon.")

    x = as_column(points.points.x)
    y = as_column(points.points.y)

    poly_offsets = as_column(polygons.polygons.part_offset)
    ring_offsets = as_column(polygons.polygons.ring_offset)
    px = as_column(polygons.polygons.x)
    py = as_column(polygons.polygons.y)

    result = cpp_point_in_polygon(x, y, poly_offsets, ring_offsets, px, py)
    result = DataFrame(
        pip_bitmap_column_to_binary_array(
            polygon_bitmap_column=result, width=len(poly_offsets) - 1
        )
    )

    result.columns = polygons.index[::-1]
    return DataFrame._from_data(
        {
            name: result[name].astype("bool")
            for name in reversed(result.columns)
        }
    )


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
    points: GeoSeries,
    polygons: GeoSeries,
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
    points: GeoSeries
        Points used to build the quadtree
    polygons: GeoSeries
        Polygons to test against

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

    if not contains_only_points(points):
        raise ValueError(
            "`point` Geoseries must contains only point geometries."
        )
    if not contains_only_polygons(polygons):
        raise ValueError(
            "`polygons` Geoseries must contains only polygons geometries."
        )

    points_x = as_column(points.points.x)
    points_y = as_column(points.points.y)

    poly_offsets = as_column(polygons.polygons.part_offset)
    ring_offsets = as_column(polygons.polygons.ring_offset)
    poly_points_x = as_column(polygons.polygons.x)
    poly_points_y = as_column(polygons.polygons.y)

    return DataFrame._from_data(
        *spatial_join.quadtree_point_in_polygon(
            poly_quad_pairs,
            quadtree,
            point_indices._column,
            points_x,
            points_y,
            poly_offsets,
            ring_offsets,
            poly_points_x,
            poly_points_y,
        )
    )


def quadtree_point_to_nearest_linestring(
    linestring_quad_pairs,
    quadtree,
    point_indices,
    points: GeoSeries,
    linestrings: GeoSeries,
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
    points: GeoSeries
        Points to find nearest linestring for
    linestrings: GeoSeries
        Linestrings to test for

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

    if not contains_only_points(points):
        raise ValueError(
            "`point` Geoseries must contains only point geometries."
        )
    if not contains_only_linestrings(linestrings):
        raise ValueError(
            "`linestrings` Geoseries must contains only linestring geometries."
        )

    if len(linestrings.lines.part_offset) != len(
        linestrings.lines.geometry_offset
    ):
        raise ValueError("GeoSeries cannot contain multilinestrings.")

    points_x = as_column(points.points.x)
    points_y = as_column(points.points.y)

    linestring_points_x = as_column(linestrings.lines.x)
    linestring_points_y = as_column(linestrings.lines.y)
    linestring_offsets = as_column(linestrings.lines.part_offset)

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
