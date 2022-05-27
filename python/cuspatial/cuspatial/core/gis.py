# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.hausdorff import (
    directed_hausdorff_distance as cpp_directed_hausdorff_distance,
)
from cuspatial._lib.linestring_distance import (
    pairwise_linestring_distance as cpp_pairwise_linestring_distance,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial._lib.polygon_bounding_boxes import (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)
from cuspatial._lib.polyline_bounding_boxes import (
    polyline_bounding_boxes as cpp_polyline_bounding_boxes,
)
from cuspatial._lib.spatial import (
    haversine_distance as cpp_haversine_distance,
    lonlat_to_cartesian as cpp_lonlat_to_cartesian,
)
from cuspatial.utils import gis_utils
from cuspatial.utils.column_utils import normalize_point_columns


def directed_hausdorff_distance(xs, ys, space_offsets):
    """Compute the directed Hausdorff distances between all pairs of
    spaces.

    Parameters
    ----------
    xs
        column of x-coordinates
    ys
        column of y-coordinates
    space_offsets
        beginning index of each space, plus the last space's end offset.

    Returns
    -------
    result : cudf.DataFrame
        The pairwise directed distance matrix with one row and one
        column per input space; the value at row i, column j represents the
        hausdorff distance from space i to space j.

    Examples
    --------
    The directed Hausdorff distance from one space to another is the greatest
    of all the distances between any point in the first space to the closest
    point in the second.

    `Wikipedia <https://en.wikipedia.org/wiki/Hausdorff_distance>`_

    Consider a pair of lines on a grid::

             :
             x
        -----xyy---
             :
             :

    x\\ :sub:`0` = (0, 0), x\\ :sub:`1` = (0, 1)

    y\\ :sub:`0` = (1, 0), y\\ :sub:`1` = (2, 0)

    x\\ :sub:`0` is the closest point in ``x`` to ``y``. The distance from
    x\\ :sub:`0` to the farthest point in ``y`` is 2.

    y\\ :sub:`0` is the closest point in ``y`` to ``x``. The distance from
    y\\ :sub:`0` to the farthest point in ``x`` is 1.414.

    Compute the directed hausdorff distances between a set of spaces

    >>> result = cuspatial.directed_hausdorff_distance(
            [0, 1, 0, 0], # xs
            [0, 0, 1, 2], # ys
            [0, 2, 4],    # space_offsets
        )
    >>> print(result)
             0         1
        0  0.0  1.414214
        1  2.0  0.000000
    """

    num_spaces = len(space_offsets)
    if num_spaces == 0:
        return DataFrame()
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    result = cpp_directed_hausdorff_distance(
        xs,
        ys,
        as_column(space_offsets, dtype="uint32"),
    )
    result = result.data_array_view
    result = result.reshape(num_spaces, num_spaces)
    return DataFrame(result)


def haversine_distance(p1_lon, p1_lat, p2_lon, p2_lat):
    """Compute the haversine distances in kilometers between an arbitrary
    list of lon/lat pairs

    Parameters
    ----------
    p1_lon
        longitude of first set of coords
    p1_lat
        latitude of first set of coords
    p2_lon
        longitude of second set of coords
    p2_lat
        latitude of second set of coords

    Returns
    -------
    result : cudf.Series
        The distance between all pairs of lon/lat coordinates
    """

    p1_lon, p1_lat, p2_lon, p2_lat = normalize_point_columns(
        as_column(p1_lon),
        as_column(p1_lat),
        as_column(p2_lon),
        as_column(p2_lat),
    )
    return cpp_haversine_distance(p1_lon, p1_lat, p2_lon, p2_lat)


def lonlat_to_cartesian(origin_lon, origin_lat, input_lon, input_lat):
    """
    Convert lon/lat to ``x,y`` coordinates with respect to an origin lon/lat
    point. Results are scaled relative to the size of the Earth in kilometers.

    Parameters
    ----------
    origin_lon : ``number``
        longitude offset  (this is subtracted from each input before
        converting to x,y)
    origin_lat : ``number``
        latitude offset (this is subtracted from each input before
        converting to x,y)
    input_lon : ``Series`` or ``list``
        longitude coordinates to convert to x
    input_lat : ``Series`` or ``list``
        latitude coordinates to convert to y

    Returns
    -------
    result : cudf.DataFrame
        x : cudf.Series
            x-coordinate of the input relative to the size of the Earth in
            kilometers.
        y : cudf.Series
            y-coordinate of the input relative to the size of the Earth in
            kilometers.
    """
    result = cpp_lonlat_to_cartesian(
        origin_lon, origin_lat, input_lon._column, input_lat._column
    )
    return DataFrame({"x": result[0], "y": result[1]})


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


def polygon_bounding_boxes(poly_offsets, ring_offsets, xs, ys):
    """Compute the minimum bounding-boxes for a set of polygons.

    Parameters
    ----------
    poly_offsets
        Begin indices of the first ring in each polygon (i.e. prefix-sum)
    ring_offsets
        Begin indices of the first point in each ring (i.e. prefix-sum)
    xs
        Polygon point x-coordinates
    ys
        Polygon point y-coordinates

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each polygon

        x_min : cudf.Series
            the minimum x-coordinate of each bounding box
        y_min : cudf.Series
            the minimum y-coordinate of each bounding box
        x_max : cudf.Series
            the maximum x-coordinate of each bounding box
        y_max : cudf.Series
            the maximum y-coordinate of each bounding box
    """
    poly_offsets = as_column(poly_offsets, dtype="int32")
    ring_offsets = as_column(ring_offsets, dtype="int32")
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *cpp_polygon_bounding_boxes(poly_offsets, ring_offsets, xs, ys)
    )


def polyline_bounding_boxes(poly_offsets, xs, ys, expansion_radius):
    """Compute the minimum bounding-boxes for a set of polylines.

    Parameters
    ----------
    poly_offsets
        Begin indices of the first ring in each polyline (i.e. prefix-sum)
    xs
        Polyline point x-coordinates
    ys
        Polyline point y-coordinates
    expansion_radius
        radius of each polyline point

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each polyline

        x_min : cudf.Series
            the minimum x-coordinate of each bounding box
        y_min : cudf.Series
            the minimum y-coordinate of each bounding box
        x_max : cudf.Series
            the maximum x-coordinate of each bounding box
        y_max : cudf.Series
            the maximum y-coordinate of each bounding box
    """
    poly_offsets = as_column(poly_offsets, dtype="int32")
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *cpp_polyline_bounding_boxes(poly_offsets, xs, ys, expansion_radius)
    )


def pairwise_linestring_distance(offsets1, xs1, ys1, offsets2, xs2, ys2):
    """Compute shortest distance between pairs of linestrings (a.k.a. polylines)

    The shortest distance between two linestrings is defined as the shortest distance
    between all pairs of segments of the two linestrings. If any of the segments intersect,
    the distance is 0.

    Parameters
    ----------
    offsets1
        Indices of the first point of the first linestring of each pair.
    xs1
        x-components of points in the first linestring of each pair.
    ys1
        y-component of points in the first linestring of each pair.
    offsets2
        Indices of the first point of the second linestring of each pair.
    xs2
        x-component of points in the second linestring of each pair.
    ys2
        y-component of points in the second linestring of each pair.

    Returns
    -------
    distance : cudf.Series
        the distance between each pair of linestrings

    Examples
    --------
    The following example contains 4 pairs of linestrings.
 
    First pair::

        (0, 1) -> (1, 0) -> (-1, 0)
        (1, 1) -> (2, 1) -> (2, 0) -> (3, 0)

            |
            *   #---#
            | \     |
        ----O---*---#---#
            | /
            *
            |

    The shortest distance between the two linestrings is the distance
    from point ``(1, 1)`` to segment ``(0, 1) -> (1, 0)``, which is
    ``sqrt(2)/2``.

    Second pair::

        (0, 0) -> (0, 1)
        (1, 0) -> (1, 1) -> (1, 2)


    These linestrings are parallel. Their distance is 1 (point
    ``(0, 0)`` to point ``(1, 0)``).
    
    Third pair::

        (0, 0) -> (2, 2) -> (-2, 0)
        (2, 0) -> (0, 2)
    

    These linestrings intersect, so their distance is 0.
    
    Forth pair::

        (2, 2) -> (-2, -2)
        (1, 1) -> (5, 5) -> (10, 0)


    These linestrings contain colinear and overlapping sections, so
    their distance is 0.
    
    The input of above example is::

        linestring1_offsets:  {0, 3, 5, 8}
        linestring1_points_x: {0, 1, -1, 0, 0, 0, 2, -2, 2, -2}
        linestring1_points_y: {1, 0, 0, 0, 1, 0, 2, 0, 2, -2}
        linestring2_offsets:  {0, 4, 7, 9}
        linestring2_points_x: {1, 2, 2, 3, 1, 1, 1, 2, 0, 1, 5, 10}
        linestring2_points_y: {1, 1, 0, 0, 0, 1, 2, 0, 2, 1, 5, 0}
        
        Result: {sqrt(2.0)/2, 1, 0, 0}
    """
    xs1, ys1, xs2, ys2 = normalize_point_columns(
        as_column(xs1), as_column(ys1), as_column(xs2), as_column(ys2)
    )
    offsets1 = as_column(offsets1, dtype="int32")
    offsets2 = as_column(offsets2, dtype="int32")
    return Series._from_data(
        {
            None: cpp_pairwise_linestring_distance(
                offsets1, xs1, ys1, offsets2, xs2, ys2
            )
        }
    )
