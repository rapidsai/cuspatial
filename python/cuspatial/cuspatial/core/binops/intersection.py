# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from typing import TYPE_CHECKING

import numpy as np

import cudf
from cudf.core.column import ColumnBase, ListColumn, as_column

from cuspatial._lib.intersection import (
    pairwise_linestring_intersection as c_pairwise_linestring_intersection,
)
from cuspatial._lib.types import GeometryType
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core._column.geometa import Feature_Enum, GeoMeta
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    empty_geometry_column,
)

if TYPE_CHECKING:
    from cuspatial.core.geoseries import GeoSeries


def pairwise_linestring_intersection(
    linestrings1: "GeoSeries", linestrings2: "GeoSeries"
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
        - An integral cuDF series of offsets to each intersection result in the
          GeoSeries.
        - A Geoseries of the results of the intersection.
        - A DataFrame of the ids of the linestrings and segments that
          the intersection results came from.
    """

    from cuspatial.core.geoseries import GeoSeries

    if len(linestrings1) == 0 and len(linestrings2) == 0:
        return (
            cudf.Series([0], dtype="i4"),
            GeoSeries([]),
            cudf.DataFrame(
                {
                    "lhs_linestring_id": [],
                    "lhs_segment_id": [],
                    "rhs_linestring_id": [],
                    "rhs_segment_id": [],
                },
                dtype="i4",
            ),
        )

    if any(
        not contains_only_linestrings(s) for s in [linestrings1, linestrings2]
    ):
        raise ValueError("Input GeoSeries must contain only linestrings.")

    geoms, look_back_ids = c_pairwise_linestring_intersection(
        linestrings1.lines.column().to_pylibcudf(mode="read"),
        linestrings2.lines.column().to_pylibcudf(mode="read"),
    )

    (
        geometry_collection_offset,
        types_buffer,
        offset_buffer,
        points,
        segments,
    ) = geoms
    geometry_collection_offset = ColumnBase.from_pylibcudf(
        geometry_collection_offset
    )
    types_buffer = ColumnBase.from_pylibcudf(types_buffer)
    # Map linestring type codes from libcuspatial to cuspatial
    types_buffer[
        types_buffer == GeometryType.LINESTRING.value
    ] = Feature_Enum.LINESTRING.value
    offset_buffer = ColumnBase.from_pylibcudf(offset_buffer)
    points = ColumnBase.from_pylibcudf(points)
    segments = ColumnBase.from_pylibcudf(segments)

    # Organize the look back ids into list column
    def id_from_pylibcudf(id_, offset_col):
        col = ColumnBase.from_pylibcudf(id_)
        return ListColumn(
            data=None,
            dtype=cudf.ListDtype(col.dtype),
            size=len(offset_col) - 1,
            children=(offset_col, col),
        )

    lhs_linestring_id, lhs_segment_id, rhs_linestring_id, rhs_segment_id = [
        id_from_pylibcudf(id_, geometry_collection_offset)
        for id_ in look_back_ids
    ]

    linestring_column = ListColumn(
        data=None,
        dtype=cudf.ListDtype(segments.dtype),
        size=segments.size,
        children=(
            as_column(range(0, len(segments) + 1), dtype=np.dtype(np.int32)),
            segments,
        ),
    )

    coord_dtype = points.dtype.leaf_type

    meta = GeoMeta(
        {"input_types": types_buffer, "union_offsets": offset_buffer}
    )
    from cuspatial.core.geoseries import GeoSeries

    geometries = GeoSeries._from_column(
        GeoColumn(
            (
                cudf.Series._from_column(points),
                cudf.Series._from_column(
                    empty_geometry_column(Feature_Enum.MULTIPOINT, coord_dtype)
                ),
                cudf.Series._from_column(linestring_column),
                cudf.Series._from_column(
                    empty_geometry_column(Feature_Enum.POLYGON, coord_dtype)
                ),
            ),
            meta,
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
