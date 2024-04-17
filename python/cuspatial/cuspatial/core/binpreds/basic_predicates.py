# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import cudf

from cuspatial.core.binops.equals_count import pairwise_multipoint_equals_count
from cuspatial.utils.binpred_utils import (
    _linestrings_from_geometry,
    _multipoints_from_geometry,
    _multipoints_is_degenerate,
    _points_and_lines_to_multipoints,
    _zero_series,
)


def _basic_equals_any(lhs, rhs):
    """Utility method that returns True if any point in the lhs geometry
    is equal to a point in the rhs geometry."""
    lhs = _multipoints_from_geometry(lhs)
    rhs = _multipoints_from_geometry(rhs)
    result = pairwise_multipoint_equals_count(lhs, rhs)
    return result > 0


def _basic_equals_all(lhs, rhs):
    """Utility method that returns True if all points in the lhs geometry
    are equal to points in the rhs geometry."""
    lhs = _multipoints_from_geometry(lhs)
    rhs = _multipoints_from_geometry(rhs)
    result = pairwise_multipoint_equals_count(lhs, rhs)
    sizes = (
        lhs.multipoints.geometry_offset[1:]
        - lhs.multipoints.geometry_offset[:-1]
    )
    return result == sizes


def _basic_equals_count(lhs, rhs):
    """Utility method that returns the number of points in the lhs geometry
    that are equal to a point in the rhs geometry."""
    lhs = _multipoints_from_geometry(lhs)
    rhs = _multipoints_from_geometry(rhs)
    result = pairwise_multipoint_equals_count(lhs, rhs)
    return result


def _basic_intersects_pli(lhs, rhs):
    """Utility method that returns the original results of
    `pairwise_linestring_intersection` (pli)."""
    from cuspatial.core.binops.intersection import (
        pairwise_linestring_intersection,
    )

    lhs = _linestrings_from_geometry(lhs)
    rhs = _linestrings_from_geometry(rhs)
    return pairwise_linestring_intersection(lhs, rhs)


def _basic_intersects_count(lhs, rhs):
    """Utility method that returns the number of points in the lhs geometry
    that intersect with the rhs geometry."""
    pli = _basic_intersects_pli(lhs, rhs)
    if len(pli[1]) == 0:
        return _zero_series(len(rhs))
    intersections = _points_and_lines_to_multipoints(pli[1], pli[0])
    sizes = cudf.Series(intersections.sizes)
    # If the result is degenerate
    is_degenerate = _multipoints_is_degenerate(intersections)
    # If all the points in the intersection are in the rhs
    if len(is_degenerate) > 0:
        sizes[is_degenerate] = sizes.dtype.type(1)
    return sizes


def _basic_intersects(lhs, rhs):
    """Utility method that returns True if any point in the lhs geometry
    intersects with the rhs geometry."""
    is_sizes = _basic_intersects_count(lhs, rhs)
    return is_sizes > 0


def _basic_contains_count(lhs, rhs):
    """Utility method that returns the number of points in the lhs geometry
    that are contained_properly in the rhs geometry.
    """
    lhs = lhs
    rhs = _multipoints_from_geometry(rhs)
    contains = lhs.contains_properly(rhs, mode="basic_count")
    return contains


def _basic_contains_any(lhs, rhs):
    """Utility method that returns True if any point in the lhs geometry
    is contained_properly in the rhs geometry."""
    lhs = lhs
    rhs = _multipoints_from_geometry(rhs)
    contains = lhs.contains_properly(rhs, mode="basic_any")
    intersects = _basic_intersects(lhs, rhs)
    return contains | intersects


def _basic_contains_properly_any(lhs, rhs):
    """Utility method that returns True if any point in the lhs geometry
    is contained_properly in the rhs geometry."""
    lhs = lhs
    rhs = _multipoints_from_geometry(rhs)
    contains = lhs.contains_properly(rhs, mode="basic_any")
    return contains
