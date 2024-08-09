# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import cupy as cp
import numpy as np

import cudf
import cudf.core.column

import cuspatial
from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core._column.geometa import Feature_Enum

"""Column-Type objects to use for simple syntax in the `DispatchDict` contained
in each `feature_<predicate>.py` file.  For example, instead of writing out
`ColumnType.POINT`, we can just write `Point`.
"""
Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON


def _false_series(size):
    """Return a Series of False values"""
    return cudf.Series(cp.zeros(size, dtype=cp.bool_))


def _true_series(size):
    """Return a Series of True values"""
    return cudf.Series(cp.ones(size, dtype=cp.bool_))


def _zero_series(size):
    """Return a Series of zeros"""
    return cudf.Series(cp.zeros(size, dtype=cp.int32))


def _one_series(size):
    """Return a Series of ones"""
    return cudf.Series(cp.ones(size, dtype=cp.int32))


def _count_results_in_multipoint_geometries(point_indices, point_result):
    """Count the number of points in each multipoint geometry.

    Parameters
    ----------
    point_indices : cudf.Series
        The indices of the points in the original (rhs) GeoSeries.
    point_result : cudf.DataFrame
        The result of a contains_properly call.

    Returns
    -------
    cudf.Series
        The number of points that fell within a particular polygon id.
    cudf.Series
        The number of points in each multipoint geometry.
    """
    point_indices_df = cudf.Series(
        point_indices,
        name="rhs_index",
        index=cudf.RangeIndex(len(point_indices), name="point_index"),
    ).reset_index()
    with_rhs_indices = point_result.merge(point_indices_df, on="point_index")
    # Because we are doing pairwise operations, we're only interested in the
    # results where polygon_index and rhs_index match
    pairwise_matches = with_rhs_indices[
        with_rhs_indices["polygon_index"] == with_rhs_indices["rhs_index"]
    ]
    points_grouped_by_original_polygon = pairwise_matches[
        ["point_index", "rhs_index"]
    ].drop_duplicates()
    hits = (
        points_grouped_by_original_polygon.groupby("rhs_index")
        .count()
        .sort_index()
    )
    expected_count = point_indices_df.groupby("rhs_index").count().sort_index()
    return hits, expected_count


def _linestrings_from_polygons(geoseries):
    """Converts the exterior and interior rings of a geoseries of polygons
    into a geoseries of linestrings."""
    xy = geoseries.polygons.xy
    parts = geoseries.polygons.part_offset.take(
        geoseries.polygons.geometry_offset
    )
    rings = geoseries.polygons.ring_offset
    return cuspatial.GeoSeries.from_linestrings_xy(
        xy,
        rings,
        parts,
    )


def _linestrings_from_multipoints(geoseries):
    """Converts a geoseries of multipoints into a geoseries of
    linestrings."""
    points = cudf.DataFrame(
        {
            "x": geoseries.multipoints.x.repeat(2).reset_index(drop=True),
            "y": geoseries.multipoints.y.repeat(2).reset_index(drop=True),
        }
    ).interleave_columns()
    result = cuspatial.GeoSeries.from_linestrings_xy(
        points,
        geoseries.multipoints.geometry_offset * 2,
        cp.arange(len(geoseries) + 1),
    )
    return result


def _linestrings_from_points(geoseries):
    """Converts a geoseries of points into a geoseries of linestrings.

    Linestrings converted to points are represented as a segment of
    length two, with the beginning and ending of the segment being the
    same point.

    Example
    -------
    >>> import cuspatial
    >>> from cuspatial.utils.binpred_utils import (
    ...     _linestrings_from_points
    ... )
    >>> from shapely.geometry import Point
    >>> points = cuspatial.GeoSeries([Point(0, 0), Point(1, 1)])
    >>> linestrings = _linestrings_from_points(points)
    >>> linestrings
    Out[1]:
    0    LINESTRING (0.00000 0.00000, 0.00000 0.00000)
    1    LINESTRING (1.00000 1.00000, 1.00000 1.00000)
    dtype: geometry
    """
    x = cp.repeat(geoseries.points.x, 2)
    y = cp.repeat(geoseries.points.y, 2)
    xy = cudf.DataFrame({"x": x, "y": y}).interleave_columns()
    parts = cp.arange((len(geoseries) + 1)) * 2
    geometries = cp.arange(len(geoseries) + 1)
    return cuspatial.GeoSeries.from_linestrings_xy(xy, parts, geometries)


def _linestrings_from_geometry(geoseries):
    """Wrapper function that converts any homogeneous geoseries into
    a geoseries of linestrings."""
    if geoseries.column_type == ColumnType.POINT:
        return _linestrings_from_points(geoseries)
    if geoseries.column_type == ColumnType.MULTIPOINT:
        return _linestrings_from_multipoints(geoseries)
    elif geoseries.column_type == ColumnType.LINESTRING:
        return geoseries
    elif geoseries.column_type == ColumnType.POLYGON:
        return _linestrings_from_polygons(geoseries)
    else:
        raise NotImplementedError(
            "Cannot convert type {} to linestrings".format(geoseries.type)
        )


def _multipoints_from_points(geoseries):
    """Converts a geoseries of points into a geoseries of length 1
    multipoints."""
    result = cuspatial.GeoSeries.from_multipoints_xy(
        geoseries.points.xy, cp.arange((len(geoseries) + 1))
    )
    return result


def _multipoints_from_linestrings(geoseries):
    """Converts a geoseries of linestrings into a geoseries of
    multipoints. MultiLineStrings are converted into a single multipoint."""
    xy = geoseries.lines.xy
    mpoints = geoseries.lines.part_offset.take(geoseries.lines.geometry_offset)
    return cuspatial.GeoSeries.from_multipoints_xy(xy, mpoints)


def _multipoints_from_polygons(geoseries):
    """Converts a geoseries of polygons into a geoseries of multipoints.
    All exterior and interior points become points in each multipoint object.
    """
    xy = geoseries.polygons.xy
    polygon_offsets = geoseries.polygons.ring_offset.take(
        geoseries.polygons.part_offset.take(geoseries.polygons.geometry_offset)
    )
    # Drop the endpoint from all polygons
    return cuspatial.GeoSeries.from_multipoints_xy(xy, polygon_offsets)


def _multipoints_from_geometry(geoseries):
    """Wrapper function that converts any homogeneous geoseries into
    a geoseries of multipoints."""
    if geoseries.column_type == ColumnType.POINT:
        return _multipoints_from_points(geoseries)
    elif geoseries.column_type == ColumnType.MULTIPOINT:
        return geoseries
    elif geoseries.column_type == ColumnType.LINESTRING:
        return _multipoints_from_linestrings(geoseries)
    elif geoseries.column_type == ColumnType.POLYGON:
        return _multipoints_from_polygons(geoseries)
    else:
        raise NotImplementedError(
            "Cannot convert type {} to multipoints".format(geoseries.type)
        )


def _points_from_linestrings(geoseries):
    """Convert a geoseries of linestrings into a geoseries of points.
    The length of the result is equal to the sum of the lengths of the
    linestrings in the original geoseries."""
    return cuspatial.GeoSeries.from_points_xy(geoseries.lines.xy)


def _points_from_polygons(geoseries):
    """Convert a geoseries of linestrings into a geoseries of points.
    The length of the result is equal to the sum of the lengths of the
    polygons in the original geoseries."""
    return cuspatial.GeoSeries.from_points_xy(geoseries.polygons.xy)


def _points_from_geometry(geoseries):
    """Wrapper function that converts any homogeneous geoseries into
    a geoseries of points."""
    if geoseries.column_type == ColumnType.POINT:
        return geoseries
    elif geoseries.column_type == ColumnType.LINESTRING:
        return _points_from_linestrings(geoseries)
    elif geoseries.column_type == ColumnType.POLYGON:
        return _points_from_polygons(geoseries)
    else:
        raise NotImplementedError(
            "Cannot convert type {} to points".format(geoseries.type)
        )


def _linestring_to_boundary(geoseries):
    """Convert a geoseries of linestrings to a geoseries of multipoints
    containing only the start and end of the linestrings."""
    x = geoseries.lines.x
    y = geoseries.lines.y
    starts = geoseries.lines.part_offset.take(geoseries.lines.geometry_offset)
    ends = (starts - 1)[1:]
    starts = starts[:-1]
    points_x = cudf.DataFrame(
        {
            "starts": x[starts].reset_index(drop=True),
            "ends": x[ends].reset_index(drop=True),
        }
    ).interleave_columns()
    points_y = cudf.DataFrame(
        {
            "starts": y[starts].reset_index(drop=True),
            "ends": y[ends].reset_index(drop=True),
        }
    ).interleave_columns()
    xy = cudf.DataFrame({"x": points_x, "y": points_y}).interleave_columns()
    mpoints = cp.arange(len(starts) + 1) * 2
    return cuspatial.GeoSeries.from_multipoints_xy(xy, mpoints)


def _polygon_to_boundary(geoseries):
    """Convert a geoseries of polygons to a geoseries of linestrings or
    multilinestrings containing only the exterior and interior boundaries
    of the polygons."""
    xy = geoseries.polygons.xy
    parts = geoseries.polygons.part_offset.take(
        geoseries.polygons.geometry_offset
    )
    rings = geoseries.polygons.ring_offset
    return cuspatial.GeoSeries.from_linestrings_xy(
        xy,
        rings,
        parts,
    )


def _is_complex(geoseries):
    """Returns True if the GeoSeries contains non-point features that
    need to be reconstructed after basic predicates have computed."""
    if len(geoseries.polygons.xy) > 0:
        return True
    if len(geoseries.lines.xy) > 0:
        return True
    if len(geoseries.multipoints.xy) > 0:
        return True
    return False


def _open_polygon_rings(geoseries):
    """Converts a geoseries of polygons into a geoseries of linestrings
    by opening the rings of each polygon."""
    x = geoseries.polygons.x
    y = geoseries.polygons.y
    parts = geoseries.polygons.part_offset.take(
        geoseries.polygons.geometry_offset
    )
    rings_mask = geoseries.polygons.ring_offset - 1
    rings_mask[0] = 0
    mask = _true_series(len(x))
    mask[rings_mask[1:]] = False
    x = x[mask]
    y = y[mask]
    xy = cudf.DataFrame({"x": x, "y": y}).interleave_columns()
    rings = geoseries.polygons.ring_offset - cp.arange(len(rings_mask))
    return cuspatial.GeoSeries.from_linestrings_xy(
        xy,
        rings,
        parts,
    )


def _points_and_lines_to_multipoints(geoseries, offsets):
    # TODO: This modification drops linestrings which are not used
    # in geopandas intersection results for contains.
    """Converts a geoseries of points and lines into a geoseries of
    multipoints.

    Given a geoseries of points and lines, this function will return a
    geoseries of multipoints. The multipoints will contain the points
    and lines in the same order as the original geoseries. The offsets
    parameter groups the points and lines into multipoints. The offsets
    parameter must be a list of integers that contains the offsets of
    the multipoints in the original geoseries. A group of four points
    and lines can be arranged into four sets of multipoints depending
    on the offset used:

    >>> import cuspatial
    >>> from cuspatial.utils.binpred_utils import (
    ...     _points_and_lines_to_multipoints
    ... )
    >>> from shapely.geometry import Point, LineString
    >>> mixed = cuspatial.GeoSeries([
    ...    Point(0, 0),
    ...    LineString([(1, 1), (2, 2)]),
    ...    Point(3, 3),
    ...    LineString([(4, 4), (5, 5)]),
    ... ])
    >>> offsets = [0, 4]
    >>> # Place all of the points and linestrings into a single
    >>> # multipoint
    >>> _points_and_lines_to_multipoints(mixed, offsets)
    0    MULTIPOINT (0.00000 0.00000, 1.00000, 1.0000, ...
    dtype: geometry
    >>> offsets = [0, 1, 2, 3, 4]
    >>> # Place each point and linestring into its own multipoint
    >>> _points_and_lines_to_multipoints(mixed, offsets)
    0    MULTIPOINT (0.00000 0.00000)
    1    MULTIPOINT (1.00000, 1.00000, 2.00000, 2.00000)
    2    MULTIPOINT (3.00000 3.00000)
    3    MULTIPOINT (4.00000, 4.00000, 5.00000, 5.00000)
    dtype: geometry
    >>> offsets = [0, 2, 4]
    >>> # Split the points and linestrings into two multipoints
    >>> _points_and_lines_to_multipoints(mixed, offsets)
    0    MULTIPOINT (0.00000 0.00000, 1.00000, 1.0000, ...
    1    MULTIPOINT (3.00000 3.00000, 4.00000, 4.0000, ...
    dtype: geometry
    """
    points_mask = geoseries.feature_types == Feature_Enum.POINT.value
    lines_mask = geoseries.feature_types == Feature_Enum.LINESTRING.value
    if (points_mask + lines_mask).sum() != len(geoseries):
        raise ValueError("Geoseries must contain only points and lines")
    points = geoseries[points_mask]
    lines = geoseries[lines_mask]
    points_offsets = _zero_series(len(geoseries))
    points_offsets[points_mask] = points_offsets.dtype.type(1)
    lines_series = geoseries[lines_mask]
    lines_sizes = lines_series.sizes
    xy = _zero_series(len(points.points.xy) + len(lines.lines.xy))
    sizes = _zero_series(len(geoseries))
    if (lines_sizes != 0).all():
        lines_sizes.index = points_offsets[lines_mask].index
        points_offsets[lines_mask] = lines_series.sizes.values
        sizes[lines_mask] = lines.sizes.values * 2
    sizes[points_mask] = sizes.dtype.type(2)
    # TODO Inevitable host device copy
    points_xy_mask = cp.array(np.repeat(points_mask, sizes.values_host))
    xy = xy.astype(points.points.xy.dtype)
    xy.iloc[points_xy_mask] = points.points.xy.reset_index(drop=True)
    xy.iloc[~points_xy_mask] = lines.lines.xy.reset_index(drop=True)
    collected_offsets = cudf.concat(
        [cudf.Series([0]), sizes.cumsum()]
    ).reset_index(drop=True)[offsets]
    result = cuspatial.GeoSeries.from_multipoints_xy(
        xy, collected_offsets // 2
    )
    return result


def _multipoints_is_degenerate(geoseries):
    """Only tests if the first two points are degenerate."""
    offsets = geoseries.multipoints.geometry_offset[:-1]
    sizes_mask = geoseries.sizes > 1
    x1 = geoseries.multipoints.x[offsets[sizes_mask]]
    x2 = geoseries.multipoints.x[offsets[sizes_mask] + 1]
    y1 = geoseries.multipoints.y[offsets[sizes_mask]]
    y2 = geoseries.multipoints.y[offsets[sizes_mask] + 1]
    result = _false_series(len(geoseries))
    is_degenerate = (
        x1.reset_index(drop=True) == x2.reset_index(drop=True)
    ) & (y1.reset_index(drop=True) == y2.reset_index(drop=True))
    result[sizes_mask] = is_degenerate.reset_index(drop=True)
    return result


def _pli_features_rebuild_offsets(pli, features):
    """Takes a modified pairwise_linestring_intersection (_pli) result
    (specifically the first two elements of its result tuple: `(offsets,
    GeoSeries)` and returns a new `offsets` buffer that corresponds with the
    items in the GeoSeries. The GeoSeries in this call is modified from the
    original pairwise_linestring_intersection result because it must be a
    homogeneous GeoSeries, where the result of
    pairwise-linestring-intersection is a mixed GeoSeries containing Points
    and LineStrings.

    The original offsets buffer at `pli[0]` refers to the geometry pair
    from which the feature intersections from `pli[1]` were computed. The new
    offsets buffer specifies the same thing, but now accounts for the missing
    values (either Points or LineStrings) that are present in `pli[1]` but
    missing in the `features` argument.

    See the docs for `_pli_points_to_multipoints` and
    `_pli_lines_to_multipoints` for the rest of the explanation.
    """
    in_sizes = (
        features.sizes if len(features) > 0 else _zero_series(len(pli[0]) - 1)
    )
    if isinstance(pli[0], cudf.core.column.ColumnBase):
        offsets = cudf.Series._from_column(pli[0])
    else:
        # TODO: Can be removed if pairwise_linestring_intersection
        # always returns a cudf.Series
        # in the first element
        offsets = cudf.Series(pli[0])
    offset_sizes = offsets[1:].reset_index(drop=True) - offsets[
        :-1
    ].reset_index(drop=True)
    in_indices = offset_sizes.index.repeat(offset_sizes)
    if len(in_indices) == 0:
        return _zero_series(len(pli[0]))
    # Just replacing the indices from the feature series with
    # the corresponding indices that are specified by the offsets buffer.
    # This is because cudf.Series.replace is extremely slow.
    in_sizes_df = in_sizes.reset_index()
    in_sizes_df.columns = ["index", "sizes"]
    sizes_replacement_series = cudf.Series(in_indices).reset_index()
    sizes_replacement_series.columns = ["index", "new_index"]
    new_in_sizes_index = in_sizes_df.merge(
        sizes_replacement_series, on="index"
    )
    in_sizes.index = new_in_sizes_index.sort_values("index")["new_index"]

    # Recompute the offsets for the new series
    grouped_sizes = in_sizes.groupby(level=0).sum().sort_index()
    out_sizes = _zero_series(len(pli[0]) - 1)
    out_sizes = out_sizes.astype(grouped_sizes.dtype)
    out_sizes.iloc[grouped_sizes.index] = grouped_sizes
    offsets = cudf.concat([cudf.Series([0]), out_sizes.cumsum()])
    return offsets


def _pli_points_to_multipoints(pli):
    """Takes a tuple (specifically the first two values returned by
    pairwise_linestring_intersection (pli)) with offsets in position 0 and a
    mixed GeoSeries in position 1. Extracts the points from the mixed
    GeoSeries such that one MultiPoint is returned for all of the Point rows
    that fall within each of the ranges specified by the offsets argument.

    Example
    -------
    pli = (
        cudf.Series([0, 2, 4, 6]),
        cuspatial.GeoSeries([
            Point(0, 0),
            LineString([(1, 1), (2, 2)]),
            Point(3, 3),
            Point(4, 4),
            Point(5, 5),
            LineString([(6, 6), (7, 7)]),
        ])
    )
    multipoints_from_points = _pli_points_to_multipoints(pli)
    print(multipoints_from_points)
    0                     MULTIPOINT (0.00000 0.00000)
    1    MULTIPOINT (3.00000 3.00000, 4.00000 4.00000)
    2                     MULTIPOINT (5.00000 5.00000)
    dtype: geometry
    """
    points = pli[1][pli[1].feature_types == Feature_Enum.POINT.value]
    offsets = _pli_features_rebuild_offsets(pli, points)
    xy = (
        points.points.xy
        if len(points.points.xy) > 0
        else cudf.Series([], dtype=cp.float64)
    )
    multipoints = cuspatial.GeoSeries.from_multipoints_xy(xy, offsets)
    return multipoints


def _pli_lines_to_multipoints(pli):
    """Works identically to `_pli_points_to_multipoints`, converting the mixed
    GeoSeries result of `pairwise_linestring_intersection` (pli) to a
    GeoSeries of MultiPoints using only the points contained in the
    LineString intersections from the result of
    pairwise_linestring_intersection."""
    lines = pli[1][pli[1].feature_types == Feature_Enum.LINESTRING.value]
    offsets = _pli_features_rebuild_offsets(pli, lines)
    xy = (
        lines.lines.xy
        if len(lines.lines.xy) > 0
        else cudf.Series([], dtype=cp.float64)
    )
    multipoints = cuspatial.GeoSeries.from_multipoints_xy(xy, offsets)
    return multipoints
