# Copyright (c) 2023, NVIDIA CORPORATION.

import cupy as cp

import cudf

import cuspatial
from cuspatial.core._column.geocolumn import ColumnType

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
    points_grouped_by_original_polygon = with_rhs_indices[
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
