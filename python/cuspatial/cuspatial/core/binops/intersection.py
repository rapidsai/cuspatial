# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf
from cudf.core.column import arange, build_list_column

from cuspatial._lib.intersection import (
    pairwise_linestring_intersection as c_pairwise_linestring_intersection,
)
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core._column.geometa import Feature_Enum
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    empty_geometry_column,
)


def pairwise_linestring_intersection(
    linestrings1: GeoSeries, linestrings2: GeoSeries
):
    """
    Compute the intersection of two GeoSeries of linestrings.

    Note
    ----
    The result contains an index list and a GeoSeries and is interpreted
    as `List<Union>`. This is a temporary workaround until cuDF supports union
    column.

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
        points,
        segments,
    ) = geoms

    # Organize the look back ids into list column
    (lhs_linestring_id, lhs_segment_id, rhs_linestring_id, rhs_segment_id,) = [
        build_list_column(
            indices=geometry_collection_offset,
            elements=id_,
            size=len(geometry_collection_offset) - 1,
        )
        for id_ in look_back_ids
    ]

    linestring_column = build_list_column(
        indices=arange(0, len(segments) + 1, dtype="int32"),
        elements=segments,
        size=len(segments),
    )

    coord_dtype = points.dtype.leaf_type
    geometries = GeoSeries(
        GeoColumn._from_arrays(
            types_buffer,
            offset_buffer,
            cudf.Series(points),
            empty_geometry_column(Feature_Enum.MULTIPOINT, coord_dtype),
            cudf.Series(linestring_column),
            empty_geometry_column(Feature_Enum.POLYGON, coord_dtype),
        )
    )

    ids = cudf.DataFrame(
        {
            "lhs_linestring_id": lhs_linestring_id,
            "lhs_segment_id": lhs_segment_id,
            "rhs_linestring_id": rhs_linestring_id,
            "rhs_segment_id": rhs_segment_id,
        }
    )
    return geometry_collection_offset, geometries, ids
