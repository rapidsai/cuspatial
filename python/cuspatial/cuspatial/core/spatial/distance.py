# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.hausdorff import (
    directed_hausdorff_distance as cpp_directed_hausdorff_distance,
)
from cuspatial._lib.linestring_distance import (
    pairwise_linestring_distance as cpp_pairwise_linestring_distance,
)
from cuspatial._lib.point_linestring_distance import (
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
)
from cuspatial._lib.spatial import haversine_distance as cpp_haversine_distance
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_points,
    normalize_point_columns,
)


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
            [0, 2],    # space_offsets
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


def pairwise_linestring_distance(offsets1, xs1, ys1, offsets2, xs2, ys2):
    """Compute shortest distance between pairs of linestrings (a.k.a. polylines)

    The shortest distance between two linestrings is defined as the shortest
    distance between all pairs of segments of the two linestrings. If any of
    the segments intersect, the distance is 0.

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
            *   #####
            | *     #
        ----O---*---#####
            | *
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


def pairwise_point_linestring_distance(
    points: GeoSeries, linestrings: GeoSeries
):
    """Compute distance between pairs of (multi)points and (multi)linestrings

    The distance between a (multi)point and a (multi)linestring
    is defined as the shortest distance between every point in the
    multipoint and every line segment in the multilinestring.

    This algorithm computes distance pairwise. The ith row in the result is
    the distance between the ith (multi)point in `points` and the ith
    (multi)linestring in `linestrings`.

    Parameters
    ----------
    points : GeoSeries
        The (multi)points to compute the distance from.
    linestrings : GeoSeries
        The (multi)linestrings to compute the distance from.

    Returns
    -------
    distance : cudf.Series

    Notes
    -----
    The input `GeoSeries` must contain a single type geometry.
    For example, `points` series cannot contain both points and polygons.
    Currently, it is unsupported that `points` contains both points and
    multipoints.

    Examples
    --------

    **Compute distances between point array to linestring arrays**

    >>> from shapely.geometry import (
    ...     Point, MultiPoint, LineString, MultiLineString
    ... )
    >>> import geopandas as gpd, cuspatial
    >>> pts = cuspatial.from_geopandas(gpd.GeoSeries([
    ...     Point(0.0, 0.0), Point(1.0, 1.0), Point(2.0, 2.0)
    ... ]))
    >>> mlines = cuspatial.from_geopandas(gpd.GeoSeries(
    ...     [
    ...        LineString([Point(-1.0, 0.0),
    ...                    Point(-0.5, -0.5),
    ...                    Point(-1.0, -0.5),
    ...                    Point(-0.5, -1.0)]),
    ...        LineString([Point(8.0, 10.0),
    ...                    Point(11.21, 9.48),
    ...                    Point(7.1, 12.5)]),
    ...        LineString([Point(1.0, 0.0), Point(0.0, 1.0)]),
    ...     ]))
    >>> cuspatial.pairwise_point_linestring_distance(pts, mlines)
    0     0.707107
    1    11.401754
    2     2.121320
    dtype: float64

    **Compute distances between multipoint to multilinestring arrays**
    >>> # 3 pairs of multi points containing 3 points each
    >>> ptsdata = [
    ...         [[9, 7], [0, 6], [7, 2]],
    ...         [[5, 8], [5, 7], [6, 0]],
    ...         [[8, 8], [6, 7], [4, 1]],
    ...     ]
    >>> # 3 pairs of multi linestrings containing 2 linestrings each
    >>> linesdata = [
    ...         [
    ...             [[86, 47], [31, 17], [84, 16], [14, 63]],
    ...             [[14, 36], [90, 73], [72, 66], [0, 5]],
    ...         ],
    ...         [
    ...             [[36, 90], [29, 31], [91, 70], [25, 78]],
    ...             [[61, 64], [89, 20], [94, 46], [37, 44]],
    ...         ],
    ...         [
    ...             [[60, 76], [29, 60], [53, 87], [8, 18]],
    ...             [[0, 16], [79, 14], [3, 6], [98, 89]],
    ...         ],
    ...     ]
    >>> pts = cuspatial.from_geopandas(
    ...     gpd.GeoSeries(map(MultiPoint, ptsdata))
    ... )
    >>> lines = cuspatial.from_geopandas(
    ...     gpd.GeoSeries(map(MultiLineString, linesdata))
    ... )
    >>> cuspatial.pairwise_point_linestring_distance(pts, lines)
    0     0.762984
    1    33.241540
    2     0.680451
    dtype: float64
    """
    if not contains_only_points(points):
        raise ValueError("`points` array must contain only points")

    if not contains_only_linestrings(linestrings):
        raise ValueError("`linestrings` array must contain only linestrings")

    if len(points.points.xy) > 0 and len(points.multipoints.xy) > 0:
        raise NotImplementedError(
            "Mixing point and multipoint geometries is not supported"
        )

    points_xy = (
        points.points.xy
        if len(points.points.xy) > 0
        else points.multipoints.xy
    )
    points_geometry_offset = (
        None
        if len(points.points.xy) > 0
        else points.multipoints.geometry_offset._column
    )

    return Series._from_data(
        {
            None: c_pairwise_point_linestring_distance(
                points_xy._column,
                linestrings.lines.part_offset._column,
                linestrings.lines.xy._column,
                points_geometry_offset,
                linestrings.lines.geometry_offset._column,
            )
        }
    )
