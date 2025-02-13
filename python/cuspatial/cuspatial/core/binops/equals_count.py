# Copyright (c) 2023-2025, NVIDIA CORPORATION.

import cudf
from cudf.core.column import ColumnBase

from cuspatial._lib.pairwise_multipoint_equals_count import (
    pairwise_multipoint_equals_count as c_pairwise_multipoint_equals_count,
)
from cuspatial.utils.column_utils import contains_only_multipoints


def pairwise_multipoint_equals_count(lhs, rhs):
    """Compute the number of points in each multipoint in the lhs that exist
    in the corresponding multipoint in the rhs.

    For each point in a multipoint in the lhs, search the
    corresponding multipoint in the rhs for a point that is equal to the
    point in the lhs. If a point is found, increment the count for that
    multipoint in the lhs.

    Parameters
    ----------
    lhs : GeoSeries
        A GeoSeries of multipoints.
    rhs : GeoSeries
        A GeoSeries of multipoints.

    Examples
    --------
    >>> import cuspatial
    >>> from shapely.geometry import MultiPoint
    >>> p1 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    >>> p2 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    >>> cuspatial.pairwise_multipoint_equals_count(p1, p2)
    0    1
    dtype: uint32

    >>> p1 = cuspatial.GeoSeries([MultiPoint([Point(0, 0)])])
    >>> p2 = cuspatial.GeoSeries([MultiPoint([Point(1, 1)])])
    >>> cuspatial.pairwise_multipoint_equals_count(p1, p2)
    0    0
    dtype: uint32

    >>> p1 = cuspatial.GeoSeries(
    ...     [
    ...         MultiPoint([Point(0, 0)]),
    ...         MultiPoint([Point(3, 3)]),
    ...         MultiPoint([Point(2, 2)]),
    ...     ]
    ... )
    >>> p2 = cuspatial.GeoSeries(
    ...     [
    ...         MultiPoint([Point(2, 2), Point(0, 0), Point(1, 1)]),
    ...         MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)]),
    ...         MultiPoint([Point(1, 1), Point(2, 2), Point(0, 0)]),
    ...     ]
    ... )
    >>> cuspatial.pairwise_multipoint_equals_count(p1, p2)
    0    1
    1    0
    2    1
    dtype: uint32

    Returns
    -------
    count : cudf.Series
        A Series of the number of points in each multipoint in the lhs that
        are equal to points in the corresponding multipoint in the rhs.
    """
    if len(lhs) == 0:
        return cudf.Series([])

    if any(not contains_only_multipoints(s) for s in [lhs, rhs]):
        raise ValueError("Input GeoSeries must contain only multipoints.")

    lhs_column = lhs._column.mpoints._column.to_pylibcudf(mode="read")
    rhs_column = rhs._column.mpoints._column.to_pylibcudf(mode="read")
    result = c_pairwise_multipoint_equals_count(lhs_column, rhs_column)

    return cudf.Series._from_column(ColumnBase.from_pylibcudf(result))
