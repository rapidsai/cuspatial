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
