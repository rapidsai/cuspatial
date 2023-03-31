# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf

from cuspatial._lib.allpairs_multipoint_equals_count import (
    allpairs_multipoint_equals_count as c_allpairs_multipoint_equals_count,
)
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import contains_only_multipoints


def allpairs_multipoint_equals_count(lhs: GeoSeries, rhs: GeoSeries):
    """
    Compute the count of times that each multipoint in the first GeoSeries
    equals each multipoint in the second GeoSeries.

    Parameters
    ----------
    multipoint : GeoSeries
        A GeoSeries of multipoints.
    multipoint : GeoSeries
        A GeoSeries of multipoints.

    Returns
    -------
    count : cudf.Series
        A Series of counts of multipoint equality.
    """
    if len(lhs) == 0:
        return cudf.Series([])

    if any(not contains_only_multipoints(s) for s in [lhs, rhs]):
        raise ValueError("Input GeoSeries must contain only linestrings.")

    return c_allpairs_multipoint_equals_count(lhs._column, rhs._column)
