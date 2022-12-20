# Copyright (c) 2020, NVIDIA CORPORATION.

from typing import TypeVar

import numpy as np

from cudf.api.types import is_datetime_dtype

GeoSeries = TypeVar("GeoSeries", bound="GeoSeries")


def normalize_point_columns(*cols):
    """
    Normalize the input columns by inferring a common floating point dtype.

    If the common dtype isn't a floating point dtype, promote the common dtype
    to float64.

    Parameters
    ----------
    {params}

    Returns
    -------
    tuple : the input columns cast to the inferred common floating point dtype
    """
    dtype = np.result_type(*cols)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float32 if dtype.itemsize <= 4 else np.float64
    return (x.astype(dtype) for x in cols)


def normalize_timestamp_column(ts, fallback_dtype="datetime64[ms]"):
    """
    Normalize the input timestamp column to one of the cuDF timestamp dtypes.

    If the input column's dtype isn't an np.datetime64, cast the column to the
    supplied `fallback_dtype` parameter.

    Parameters
    ----------
    {params}

    Returns
    -------
    column : the input column
    """
    return ts if is_datetime_dtype(ts.dtype) else ts.astype(fallback_dtype)


def contain_single_type_geometry(gs: GeoSeries):
    """
    Returns true if `gs` contains only single type of geometries

    A geometry is considered as the same type to its multi-geometry variant.
    """
    has_points = len(gs.points.xy) > 0
    has_multipoints = len(gs.multipoints.xy) > 0
    has_lines = len(gs.lines.xy) > 0
    has_polygons = len(gs.polygons.xy) > 0

    return (
        len(gs) > 0
        and ((has_points or has_multipoints) + has_lines + has_polygons) == 1
    )


def contains_only_points(gs: GeoSeries):
    """
    Returns true if `gs` contains only points or multipoints
    """

    return contain_single_type_geometry(gs) and (
        len(gs.points.xy) > 0 or len(gs.multipoints.xy) > 0
    )


def contains_only_multipoints(gs: GeoSeries):
    """
    Returns true if `gs` contains only multipoints
    """

    return contain_single_type_geometry(gs) and (len(gs.multipoints.xy) > 0)


def contains_only_linestrings(gs: GeoSeries):
    """
    Returns true if `gs` contains only linestrings
    """

    return contain_single_type_geometry(gs) and len(gs.lines.xy) > 0


def contains_only_polygons(gs: GeoSeries):
    """
    Returns true if `gs` contains only polygons
    """

    return contain_single_type_geometry(gs) and len(gs.polygons.xy) > 0


def has_same_geometry(lhs: GeoSeries, rhs: GeoSeries):
    """
    Returns true if `lhs` and `rhs` have only features of the same homogeneous
    geometry type.
    """

    if contains_only_points(lhs) and contains_only_points(rhs):
        return True
    elif contains_only_multipoints(lhs) and contains_only_multipoints(rhs):
        return True
    elif contains_only_linestrings(lhs) and contains_only_linestrings(rhs):
        return True
    elif contains_only_polygons(lhs) and contains_only_polygons(rhs):
        return True
    else:
        return False
