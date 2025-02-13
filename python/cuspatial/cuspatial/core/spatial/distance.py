# Copyright (c) 2022-2025, NVIDIA CORPORATION.

import cudf
from cudf import DataFrame, Series
from cudf.core.column import ColumnBase, as_column

from cuspatial._lib.distance import (
    directed_hausdorff_distance as cpp_directed_hausdorff_distance,
    haversine_distance as cpp_haversine_distance,
    pairwise_linestring_distance as cpp_pairwise_linestring_distance,
    pairwise_linestring_polygon_distance as c_pairwise_line_poly_dist,
    pairwise_point_distance as cpp_pairwise_point_distance,
    pairwise_point_linestring_distance as c_pairwise_point_linestring_distance,
    pairwise_point_polygon_distance as c_pairwise_point_polygon_distance,
    pairwise_polygon_distance as c_pairwise_polygon_distance,
)
from cuspatial._lib.types import CollectionType
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_points,
    contains_only_polygons,
)


def directed_hausdorff_distance(multipoints: GeoSeries):
    """Compute the directed Hausdorff distances between all pairs of
    spaces.

    Parameters
    ----------
    multipoints: GeoSeries
        A column of multipoint, where each multipoint indicates an input space
        to compute its hausdorff distance to the rest of input spaces.

    Returns
    -------
    result : cudf.DataFrame
        result[i, j] indicates the hausdorff distance between multipoints[i]
        and multipoint[j].

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

    >>> pts = cuspatial.GeoSeries([
    ...     MultiPoint([(0, 0), (1, 0)]),
    ...     MultiPoint([(0, 1), (0, 2)])
    ... ])
    >>> cuspatial.directed_hausdorff_distance(pts)
        0         1
    0  0.0  1.414214
    1  2.0  0.000000
    """

    num_spaces = len(multipoints)
    if num_spaces == 0:
        return DataFrame([])

    if not contains_only_multipoints(multipoints):
        raise ValueError("Input must be a series of multipoints.")

    result = cpp_directed_hausdorff_distance(
        multipoints.multipoints.x._column.to_pylibcudf(mode="read"),
        multipoints.multipoints.y._column.to_pylibcudf(mode="read"),
        as_column(multipoints.multipoints.geometry_offset[:-1]).to_pylibcudf(
            mode="read"
        ),
    )

    # the column label of each column in result should be its int position
    # e.g. a dict of {0: result[0], 1, result[1], ...}
    return DataFrame._from_data(
        dict(enumerate(ColumnBase.from_pylibcudf(col) for col in result))
    )


def haversine_distance(p1: GeoSeries, p2: GeoSeries):
    """Compute the haversine distances in kilometers between an arbitrary
    list of lon/lat pairs

    Parameters
    ----------
    p1: GeoSeries
        Series of points as floats
    p2: GeoSeries
        Series of points as floats

    Returns
    -------
    result : cudf.Series
        The distance between pairs of points between `p1` and `p2`
    >>> import cudf
    >>> import cuspatial
    >>> a = {"latitude":[0.0,0.0,1.0,1.0],
    ...      "longitude": [0.0,1.0,0.0,1.0]}
    >>> df = cudf.DataFrame(data=a)

    >>> # Create cuSpatial GeoSeries from cuDF Dataframe
    >>> gs = cuspatial.GeoSeries.from_points_xy(
    ...     df[['longitude', 'latitude']].interleave_columns()
    ... )

    >>> # Create Comparator cuSpatial GeoSeries from a comparator point
    >>> df['compare_lat'] = 2.0 # this will broadcast the value to all rows
    >>> df['compare_lng'] = 2.0
    >>> cmp_gs = cuspatial.GeoSeries.from_points_xy(
    ...     df[['compare_lat', 'compare_lng']].interleave_columns()
    ... )

    >>> # Calculate Haversine Distance of cuDF dataframe to comparator point
    >>> df['compare_dist'] = cuspatial.haversine_distance(gs, cmp_gs)
    >>> df.head()
           latitude  longitude  compare_lat  compare_lng  compare_dist
    0       0.0        0.0          2.0          2.0    314.474805
    1       0.0        1.0          2.0          2.0    248.629315
    2       1.0        0.0          2.0          2.0    248.568719
    3       1.0        1.0          2.0          2.0    157.225432
    """

    if any([not contains_only_points(p1), not contains_only_points(p2)]):
        raise ValueError("Input muist be two series of points.")

    if len(p1) != len(p2):
        raise ValueError("Mismatch length of inputs.")

    p1_lon = p1.points.x._column
    p1_lat = p1.points.y._column
    p2_lon = p2.points.x._column
    p2_lat = p2.points.y._column

    return cudf.Series._from_column(
        ColumnBase.from_pylibcudf(
            cpp_haversine_distance(
                p1_lon.to_pylibcudf(mode="read"),
                p1_lat.to_pylibcudf(mode="read"),
                p2_lon.to_pylibcudf(mode="read"),
                p2_lat.to_pylibcudf(mode="read"),
            )
        )
    )


def pairwise_point_distance(points1: GeoSeries, points2: GeoSeries):
    """Compute distance between (multi)points-(multi)points pairs

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

    (
        lhs_column,
        lhs_point_collection_type,
    ) = _extract_point_column_and_collection_type(points1)
    (
        rhs_column,
        rhs_point_collection_type,
    ) = _extract_point_column_and_collection_type(points2)

    return Series._from_column(
        ColumnBase.from_pylibcudf(
            cpp_pairwise_point_distance(
                lhs_point_collection_type,
                rhs_point_collection_type,
                lhs_column.to_pylibcudf(mode="read"),
                rhs_column.to_pylibcudf(mode="read"),
            )
        )
    )


def pairwise_linestring_distance(
    multilinestrings1: GeoSeries, multilinestrings2: GeoSeries
):
    """Compute distance between (multi)linestring-(multi)linestring pairs

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

    if not contains_only_linestrings(
        multilinestrings1
    ) or not contains_only_linestrings(multilinestrings2):
        raise ValueError(
            "`multilinestrings1` and `multilinestrings2` must contain only "
            "linestrings"
        )

    if len(multilinestrings1) == 0:
        return cudf.Series(dtype="float64")

    return Series._from_column(
        ColumnBase.from_pylibcudf(
            cpp_pairwise_linestring_distance(
                multilinestrings1.lines.column().to_pylibcudf(mode="read"),
                multilinestrings2.lines.column().to_pylibcudf(mode="read"),
            )
        )
    )


def pairwise_point_linestring_distance(
    points: GeoSeries, linestrings: GeoSeries
):
    """Compute distance between (multi)points-(multi)linestrings pairs

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

    (
        point_column,
        point_collection_type,
    ) = _extract_point_column_and_collection_type(points)

    return Series._from_column(
        ColumnBase.from_pylibcudf(
            c_pairwise_point_linestring_distance(
                point_collection_type,
                point_column.to_pylibcudf(mode="read"),
                linestrings.lines.column().to_pylibcudf(mode="read"),
            )
        )
    )


def pairwise_point_polygon_distance(points: GeoSeries, polygons: GeoSeries):
    """Compute distance between (multi)points-(multi)polygons pairs

    The distance between a (multi)point and a (multi)polygon
    is defined as the shortest distance between every point in the
    multipoint and every edge of the (multi)polygon. If the multipoint and
    multipolygon intersects, the distance is 0.

    This algorithm computes distance pairwise. The ith row in the result is
    the distance between the ith (multi)point in `points` and the ith
    (multi)polygon in `polygons`.

    Parameters
    ----------
    points : GeoSeries
        The (multi)points to compute the distance from.
    polygons : GeoSeries
        The (multi)polygons to compute the distance from.

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
    Compute distance between a point and a polygon:
    >>> from shapely.geometry import Point
    >>> points = cuspatial.GeoSeries([Point(0, 0)])
    >>> polygons = cuspatial.GeoSeries([Point(1, 1).buffer(0.5)])
    >>> cuspatial.pairwise_point_polygon_distance(points, polygons)
    0    0.914214
    dtype: float64

    Compute distance between a multipoint and a multipolygon

    >>> from shapely.geometry import MultiPoint
    >>> mpoints = cuspatial.GeoSeries([MultiPoint([Point(0, 0), Point(1, 1)])])
    >>> mpolys = cuspatial.GeoSeries([
    ...     MultiPoint([Point(2, 2), Point(1, 2)]).buffer(0.5)])
    >>> cuspatial.pairwise_point_polygon_distance(mpoints, mpolys)
    0    0.5
    dtype: float64
    """

    if len(points) != len(polygons):
        raise ValueError("Unmatched input geoseries length.")

    if len(points) == 0:
        return cudf.Series(dtype=points.points.xy.dtype)

    if not contains_only_points(points):
        raise ValueError("`points` array must contain only points")

    if not contains_only_polygons(polygons):
        raise ValueError("`polygons` array must contain only polygons")

    if len(points.points.xy) > 0 and len(points.multipoints.xy) > 0:
        raise NotImplementedError(
            "Mixing point and multipoint geometries is not supported"
        )

    (
        points_column,
        point_collection_type,
    ) = _extract_point_column_and_collection_type(points)
    polygon_column = polygons.polygons.column()

    return Series._from_column(
        ColumnBase.from_pylibcudf(
            c_pairwise_point_polygon_distance(
                point_collection_type,
                points_column.to_pylibcudf(mode="read"),
                polygon_column.to_pylibcudf(mode="read"),
            )
        )
    )


def pairwise_linestring_polygon_distance(
    linestrings: GeoSeries, polygons: GeoSeries
):
    """Compute distance between (multi)linestrings-(multi)polygons pairs.

    The distance between a (multi)linestrings and a (multi)polygon
    is defined as the shortest distance between every segment in the
    multilinestring and every edge of the (multi)polygon. If the
    multilinestring and multipolygon intersect, the distance is 0.

    This algorithm computes distance pairwise. The ith row in the result is
    the distance between the ith (multi)linestring in `linestrings` and the ith
    (multi)polygon in `polygons`.

    Parameters
    ----------
    linestrings : GeoSeries
        The (multi)linestrings to compute the distance from.
    polygons : GeoSeries
        The (multi)polygons to compute the distance from.

    Returns
    -------
    distance : cudf.Series

    Notes
    -----
    The input `GeoSeries` must contain a single type geometry.
    For example, `linestrings` series cannot contain both linestrings and
    polygons.

    Examples
    --------
    Compute distance between a linestring and a polygon:
    >>> from shapely.geometry import LineString, Polygon
    >>> lines = cuspatial.GeoSeries([
    ...     LineString([(0, 0), (1, 1)])])
    >>> polys = cuspatial.GeoSeries([
    ...     Polygon([(-1, -1), (-1, 0), (-2, 0), (-1, -1)])
    ... ])
    >>> cuspatial.pairwise_linestring_polygon_distance(lines, polys)
    0    1.0
    dtype: float64

    Compute distance between a multipoint and a multipolygon
    >>> from shapely.geometry import MultiLineString, MultiPolygon
    >>> lines = cuspatial.GeoSeries([
    ...     MultiLineString([
    ...             LineString([(0, 0), (1, 1)]),
    ...             LineString([(1, 1), (2, 2)])])
    ...     ])
    >>> polys = cuspatial.GeoSeries([
    ...     MultiPolygon([
    ...             Polygon([(-1, -1), (-1, 0), (-2, 0), (-1, -1)]),
    ...             Polygon([(-2, 0), (-3, 0), (-3, -1), (-2, 0)])])
    ...     ])
    >>> cuspatial.pairwise_linestring_polygon_distance(lines, polys)
    0    1.0
    dtype: float64
    """

    if len(linestrings) != len(polygons):
        raise ValueError("Unmatched input geoseries length.")

    if len(linestrings) == 0:
        return cudf.Series(dtype=linestrings.lines.xy.dtype)

    if not contains_only_linestrings(linestrings):
        raise ValueError("`linestrings` array must contain only linestrings")

    if not contains_only_polygons(polygons):
        raise ValueError("`polygon` array must contain only polygons")

    linestrings_column = linestrings.lines.column().to_pylibcudf(mode="read")
    polygon_column = polygons.polygons.column().to_pylibcudf(mode="read")

    return Series._from_column(
        ColumnBase.from_pylibcudf(
            c_pairwise_line_poly_dist(linestrings_column, polygon_column)
        )
    )


def pairwise_polygon_distance(polygons1: GeoSeries, polygons2: GeoSeries):
    """Compute distance between (multi)polygon-(multi)polygon pairs.

    The distance between two (multi)polygons is defined as the shortest
    distance between any edge of the first (multi)polygon and any edge
    of the second (multi)polygon. If two (multi)polygons intersect, the
    distance is 0.

    This algorithm computes distance pairwise. The ith row in the result is
    the distance between the ith (multi)polygon in `polygons1` and the ith
    (multi)polygon in `polygons2`.

    Parameters
    ----------
    polygons1 : GeoSeries
        The (multi)polygons to compute the distance from.
    polygons2 : GeoSeries
        The (multi)polygons to compute the distance from.
    Returns
    -------
    distance : cudf.Series

    Notes
    -----
    `polygons1` and `polygons2` must contain only polygons.

    Examples
    --------
    Compute distance between polygons:

    >>> from shapely.geometry import Polygon, MultiPolygon
    >>> s0 = cuspatial.GeoSeries([
    ...     Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])])
    >>> s1 = cuspatial.GeoSeries([
    ...     Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])])
    >>> cuspatial.pairwise_polygon_distance(s0, s1)
    0    1.414214
    dtype: float64

    Compute distance between multipolygons:
    >>> s0 = cuspatial.GeoSeries([
    ...     MultiPolygon([
    ...         Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
    ...         Polygon([(2, 0), (3, 0), (3, 1), (2, 0)])])])
    >>> s1 = cuspatial.GeoSeries([
    ...     MultiPolygon([
    ...         Polygon([(-1, 0), (-2, 0), (-2, -1), (-1, -1), (-1, 0)]),
    ...         Polygon([(0, -1), (1, -1), (1, -2), (0, -2), (0, -1)])])])
    >>> cuspatial.pairwise_polygon_distance(s0, s1)
    0    1.0
    dtype: float64
    """

    if len(polygons1) != len(polygons2):
        raise ValueError("Unmatched input geoseries length.")

    if len(polygons1) == 0:
        return cudf.Series(dtype=polygons1.lines.xy.dtype)

    if not contains_only_polygons(polygons1):
        raise ValueError("`polygons1` array must contain only polygons")

    if not contains_only_polygons(polygons2):
        raise ValueError("`polygons2` array must contain only polygons")

    polygon1_column = polygons1.polygons.column().to_pylibcudf(mode="read")
    polygon2_column = polygons2.polygons.column().to_pylibcudf(mode="read")

    return Series._from_column(
        ColumnBase.from_pylibcudf(
            c_pairwise_polygon_distance(polygon1_column, polygon2_column)
        )
    )


def _extract_point_column_and_collection_type(s: GeoSeries):
    """Given a GeoSeries that contains only points or multipoints, return
    the point or multipoint column and the collection type of the GeoSeries.
    """
    point_collection_type = (
        CollectionType.SINGLE if len(s.points.xy > 0) else CollectionType.MULTI
    )

    if point_collection_type == CollectionType.SINGLE:
        return s.points.column(), point_collection_type
    else:
        return s.multipoints.column(), point_collection_type
