# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import Series
from cudf.core.column import as_column

from cuspatial._lib.pairwise_point_in_polygon import (
    pairwise_point_in_polygon as cpp_pairwise_point_in_polygon,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils.column_utils import normalize_point_columns


def contains(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
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

    if len(test_points_x) == len(poly_offsets):
        pip_result = cpp_pairwise_point_in_polygon(
            test_points_x,
            test_points_y,
            as_column(poly_offsets, dtype="int32"),
            as_column(poly_ring_offsets, dtype="int32"),
            poly_points_x,
            poly_points_y,
        )
    else:
        pip_result = cpp_point_in_polygon(
            test_points_x,
            test_points_y,
            as_column(poly_offsets, dtype="int32"),
            as_column(poly_ring_offsets, dtype="int32"),
            poly_points_x,
            poly_points_y,
        )

    result = Series(pip_result, dtype="bool")
    return result
