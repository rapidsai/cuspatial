# Copyright (c) 2022, NVIDIA CORPORATION.

import cudf
from cudf.core.column import arange, build_list_column

from cuspatial._lib.intersection import (
    pairwise_linestring_intersection as c_pairwise_linestring_intersection,
)
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import contains_only_linestrings


def pairwise_linestring_intersection(
    linestrings1: GeoSeries, linestrings2: GeoSeries
):
    """
    Compute the intersection of two GeoSeries of linestrings.

    Note
    ----
    Note that the result should be interpreted as type list<union>, since
    cuDF does not support union type, the result is returned as a sperate list
    and a GeoSeries.

    Parameters
    ----------
    linestrings1 : GeoSeries
        A GeoSeries of linestrings.
    linestrings2 : GeoSeries
        A GeoSeries of linestrings.

    Returns
    -------
    Tuple[cudf.Series, GeoSeries, DataFrame]
        A tuple of three elements:
        - An integral cudf serial to indicate the offsets of the geoseries.
        - A Geoseries of the results of the intersection.
        - A DataFrame of the ids of the linestrings and segments that
          the intersection results came from.
    """

    if len(linestrings1) == 0:
        return (
            cudf.Series([0]),
            GeoSeries([]),
            cudf.DataFrame(
                {
                    "lhs_linestring_id": [],
                    "lhs_segment_id": [],
                    "rhs_linestring_id": [],
                    "rhs_segment_id": [],
                }
            ),
        )

    if not contains_only_linestrings(
        linestrings1
    ) or not contains_only_linestrings(linestrings2):
        raise ValueError("Input GeoSeries must contain only linestrings.")

    geoms, look_back_ids = c_pairwise_linestring_intersection(
        linestrings1._column.lines._column, linestrings2._column.lines._column
    )

    (
        geometry_collection_offset,
        types_buffer,
        offset_buffer,
        points_xy,
        segments_offsets,
        segments_xy,
    ) = geoms

    (
        lhs_linestring_id,
        lhs_segment_id,
        rhs_linestring_id,
        rhs_segment_id,
    ) = _create_list_for_ids(geometry_collection_offset, look_back_ids)

    points_column = build_list_column(
        indices=arange(0, len(points_xy) + 1, 2, dtype="int32"),
        elements=points_xy,
        size=len(points_xy) // 2,
    )

    segment_column = build_list_column(
        indices=arange(0, len(segments_offsets), dtype="int32"),
        elements=build_list_column(
            indices=segments_offsets,
            elements=segments_xy,
            size=len(segments_offsets) - 1,
        ),
        size=len(segments_offsets) - 1,
    )

    geometries = GeoSeries(
        GeoColumn._from_arrays(
            types_buffer,
            offset_buffer,
            cudf.Series(points_column),
            cudf.Series(),
            cudf.Series(segment_column),
            cudf.Series(),
        )
    )

    breakpoint()

    ids = cudf.DataFrame(
        {
            "lhs_linestring_id": lhs_linestring_id,
            "lhs_segment_id": lhs_segment_id,
            "rhs_linestring_id": rhs_linestring_id,
            "rhs_segment_id": rhs_segment_id,
        }
    )
    return geometry_collection_offset, geometries, ids


def _create_list_for_ids(geometry_collection_offset, ids):
    return [
        build_list_column(
            indices=geometry_collection_offset,
            elements=id_,
            size=len(geometry_collection_offset) - 1,
        )
        for id_ in ids
    ]
