# Copyright (c) 2022, NVIDIA CORPORATION.

from typing import Tuple

import cudf
from cudf import DataFrame, Series
from cudf.core.column import as_column

from cuspatial._lib.distance import (
    pairwise_linestring_distance as cpp_pairwise_linestring_distance,
    pairwise_point_distance as cpp_pairwise_point_distance,
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
)
from cuspatial._lib.hausdorff import (
    directed_hausdorff_distance as cpp_directed_hausdorff_distance,
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


def pairwise_point_distance(points1: GeoSeries, points2: GeoSeries):
    """Compute shortest distance between pairs of points and multipoints

    Currently `points1` and `points2` must contain either only points or
    multipoints. Mixing points and multipoints in the same series is
    unsupported.

    Parameters
    ----------
    points1 : GeoSeries
        A GeoSeries of (multi)points
    points2 : GeoSeries
        A GeoSeries of (multi)points

    Returns
    -------
    distance : cudf.Series
        the distance between each pair of (multi)points

    Examples
    --------
    >>> from shapely.geometry import Point, MultiPoint
    >>> p1 = cuspatial.GeoSeries([
    ...     MultiPoint([(0.0, 0.0), (1.0, 0.0)]),
    ...     MultiPoint([(0.0, 1.0), (1.0, 0.0)])
    ... ])
    >>> p2 = cuspatial.GeoSeries([
    ...     Point(2.0, 2.0), Point(0.0, 0.5)
    ... ])
    >>> cuspatial.pairwise_point_distance(p1, p2)
    0    2.236068
    1    0.500000
    dtype: float64
    """

    if not len(points1) == len(points2):
        raise ValueError("`points1` and `points2` must have the same length")

    if len(points1) == 0:
        return cudf.Series(dtype="float64")

    if not contains_only_points(points1):
        raise ValueError("`points1` array must contain only points")
    if not contains_only_points(points2):
        raise ValueError("`points2` array must contain only points")
    if (len(points1.points.xy) > 0 and len(points1.multipoints.xy) > 0) or (
        len(points2.points.xy) > 0 and len(points2.multipoints.xy) > 0
    ):
        raise NotImplementedError(
            "Mixing point and multipoint geometries is not supported"
        )

    points1_xy, points1_geometry_offsets = _flatten_point_series(points1)
    points2_xy, points2_geometry_offsets = _flatten_point_series(points2)
    return Series._from_data(
        {
            None: cpp_pairwise_point_distance(
                points1_xy,
                points2_xy,
                points1_geometry_offsets,
                points2_geometry_offsets,
            )
        }
    )


def pairwise_linestring_distance(
    multilinestrings1: GeoSeries, multilinestrings2: GeoSeries
):
    """Compute shortest distance between pairs of linestrings

    The shortest distance between two linestrings is defined as the shortest
    distance between all pairs of segments of the two linestrings. If any of
    the segments intersect, the distance is 0.

    Parameters
    ----------
    multilinestrings1 : GeoSeries
        A GeoSeries of (multi)linestrings
    multilinestrings2 : GeoSeries
        A GeoSeries of (multi)linestrings

    Returns
    -------
    distance : cudf.Series
        the distance between each pair of linestrings

    Examples
    --------
    >>> from shapely.geometry import LineString, MultiLineString
    >>> ls1 = cuspatial.GeoSeries([
    ...     LineString([(0, 0), (1, 1)]),
    ...     LineString([(1, 0), (2, 1)])
    ... ])
    >>> ls2 = cuspatial.GeoSeries([
    ...     MultiLineString([
    ...         LineString([(-1, 0), (-2, -1)]),
    ...         LineString([(-2, -1), (-3, -2)])
    ...     ]),
    ...     MultiLineString([
    ...         LineString([(0, -1), (0, -2), (0, -3)]),
    ...         LineString([(0, -3), (-1, -3), (-2, -3)])
    ...     ])
    ... ])
    >>> cuspatial.pairwise_linestring_distance(ls1, ls2)
    0    1.000000
    1    1.414214
    dtype: float64
    """

    if not len(multilinestrings1) == len(multilinestrings2):
        raise ValueError(
            "`multilinestrings1` and `multilinestrings2` must have the same "
            "length"
        )

    if len(multilinestrings1) == 0:
        return cudf.Series(dtype="float64")

    return Series._from_data(
        {
            None: cpp_pairwise_linestring_distance(
                as_column(multilinestrings1.lines.part_offset),
                as_column(multilinestrings1.lines.xy),
                as_column(multilinestrings2.lines.part_offset),
                as_column(multilinestrings2.lines.xy),
                as_column(multilinestrings1.lines.geometry_offset),
                as_column(multilinestrings2.lines.geometry_offset),
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

    point_xy_col, points_geometry_offset = _flatten_point_series(points)

    return Series._from_data(
        {
            None: c_pairwise_point_linestring_distance(
                point_xy_col,
                as_column(linestrings.lines.part_offset),
                linestrings.lines.xy._column,
                points_geometry_offset,
                as_column(linestrings.lines.geometry_offset),
            )
        }
    )


def _flatten_point_series(
    points: GeoSeries,
) -> Tuple[
    cudf.core.column.column.ColumnBase, cudf.core.column.column.ColumnBase
]:
    """Given a geoseries of (multi)points, extract the offset and x/y column"""
    if len(points.points.xy) > 0:
        return points.points.xy._column, None
    return (
        points.multipoints.xy._column,
        as_column(points.multipoints.geometry_offset),
    )
