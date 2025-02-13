# Copyright (c) 2022-2025, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import ColumnBase, as_column

from cuspatial._lib.linestring_bounding_boxes import (
    linestring_bounding_boxes as cpp_linestring_bounding_boxes,
)
from cuspatial._lib.polygon_bounding_boxes import (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_polygons,
)


def polygon_bounding_boxes(polygons: GeoSeries):
    """Compute the minimum bounding-boxes for a set of polygons.

    Parameters
    ----------
    polygons: GeoSeries
        A series of polygons

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each polygon

        minx : cudf.Series
            the minimum x-coordinate of each bounding box
        miny : cudf.Series
            the minimum y-coordinate of each bounding box
        maxx : cudf.Series
            the maximum x-coordinate of each bounding box
        maxy : cudf.Series
            the maximum y-coordinate of each bounding box

    Notes
    -----
    Has no notion of multipolygons. If a multipolygon is passed, the bounding
    boxes for each polygon will be computed and returned. The user is
    responsible for handling the multipolygon case.
    """

    column_names = ["minx", "miny", "maxx", "maxy"]
    if len(polygons) == 0:
        return DataFrame(columns=column_names, dtype="f8")

    if not contains_only_polygons(polygons):
        raise ValueError("Geoseries must contain only polygons.")

    # `cpp_polygon_bounding_boxes` computes bbox with all points supplied,
    # but only supports single-polygon input. We compute the polygon offset
    # by combining the geometry offset and parts offset of the multipolygon
    # array.

    poly_offsets = polygons.polygons.part_offset
    ring_offsets = polygons.polygons.ring_offset
    x = polygons.polygons.x
    y = polygons.polygons.y

    return DataFrame._from_data(
        dict(
            zip(
                column_names,
                (
                    ColumnBase.from_pylibcudf(col)
                    for col in cpp_polygon_bounding_boxes(
                        as_column(poly_offsets).to_pylibcudf(mode="read"),
                        as_column(ring_offsets).to_pylibcudf(mode="read"),
                        as_column(x).to_pylibcudf(mode="read"),
                        as_column(y).to_pylibcudf(mode="read"),
                    )
                ),
            )
        )
    )


def linestring_bounding_boxes(linestrings: GeoSeries, expansion_radius: float):
    """Compute the minimum bounding boxes for a set of linestrings.

    Parameters
    ----------
    linestrings: GeoSeries
        A series of linestrings
    expansion_radius
        radius of each linestring point

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each linestring

        minx : cudf.Series
            the minimum x-coordinate of each bounding box
        miny : cudf.Series
            the minimum y-coordinate of each bounding box
        maxx : cudf.Series
            the maximum x-coordinate of each bounding box
        maxy : cudf.Series
            the maximum y-coordinate of each bounding box
    """

    column_names = ["minx", "miny", "maxx", "maxy"]
    if len(linestrings) == 0:
        return DataFrame(columns=column_names, dtype="f8")

    if not contains_only_linestrings(linestrings):
        raise ValueError("Geoseries must contain only linestrings.")

    # `cpp_linestring_bounding_boxes` computes bbox with all points supplied,
    # but only supports single-linestring input. We compute the linestring
    # offset by combining the geometry offset and parts offset of the
    # multilinestring array.

    line_offsets = linestrings.lines.part_offset.take(
        linestrings.lines.geometry_offset
    )
    x = linestrings.lines.x
    y = linestrings.lines.y

    results = cpp_linestring_bounding_boxes(
        as_column(line_offsets).to_pylibcudf(mode="read"),
        as_column(x).to_pylibcudf(mode="read"),
        as_column(y).to_pylibcudf(mode="read"),
        expansion_radius,
    )

    return DataFrame._from_data(
        dict(
            zip(
                column_names,
                (ColumnBase.from_pylibcudf(col) for col in results),
            )
        )
    )
