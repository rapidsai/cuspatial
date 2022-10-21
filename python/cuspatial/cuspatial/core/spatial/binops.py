# Copyright (c) 2022, NVIDIA CORPORATION.

import cupy as cp

from cudf import Series
from cudf.core.column import as_column

from cuspatial._lib.contains import contains as cpp_contains
from cuspatial.utils import gis_utils
from cuspatial.utils.column_utils import normalize_point_columns


def contains(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """Compute from a set of points and a set of polygons which points fall
    within each polygon. Note that `polygons_(x,y)` must be specified as
    closed polygons: the first and last coordinate of each polygon must be
    the same.

    Parameters
    ----------
    test_points_x
        x-coordinate of test points
    test_points_y
        y-coordinate of test points
    poly_offsets
        beginning index of the first ring in each polygon
    poly_ring_offsets
        beginning index of the first point in each ring
    poly_points_x
        x closed-coordinate of polygon points
    poly_points_y
        y closed-coordinate of polygon points

    Examples
    --------

    Test whether 3 points fall within either of two polygons

    # TODO: Examples

    note
    input Series x and y will not be index aligned, but computed as
    sequential arrays.

    note
    poly_ring_offsets must contain only the rings that make up the polygons
    indexed by poly_offsets. If there are rings in poly_ring_offsets that
    are not part of the polygons in poly_offsets, results are likely to be
    incorrect and behavior is undefined.

    Returns
    -------
    result : cudf.DataFrame
        A DataFrame of boolean values indicating whether each point falls
        within each polygon.
    """

    if len(poly_offsets) == 0:
        return Series()

    (
        test_points_x,
        test_points_y,
        poly_points_x,
        poly_points_y,
    ) = normalize_point_columns(
        as_column(test_points_x),
        as_column(test_points_y),
        as_column(poly_points_x),
        as_column(poly_points_y),
    )

    contains_bitmap = cpp_contains(
        test_points_x,
        test_points_y,
        as_column(poly_offsets, dtype="int32"),
        as_column(poly_ring_offsets, dtype="int32"),
        poly_points_x,
        poly_points_y,
    )

    # TODO: Should be able to make these changes at the C++ level instead
    # of here.
    to_bool = gis_utils.pip_bitmap_column_to_binary_array(
        polygon_bitmap_column=contains_bitmap, width=len(poly_offsets)
    )
    flattened = (to_bool[::-1] * cp.identity(len(poly_offsets))).diagonal()
    result = Series(flattened, dtype="bool")
    breakpoint()
    return result
