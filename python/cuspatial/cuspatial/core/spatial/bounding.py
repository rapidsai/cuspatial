# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib.linestring_bounding_boxes import (
    linestring_bounding_boxes as cpp_linestring_bounding_boxes,
)
from cuspatial._lib.polygon_bounding_boxes import (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_polygons,
    normalize_point_columns,
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
    """

    column_names = ["minx", "miny", "maxx", "maxy"]
    if len(polygons) == 0:
        return DataFrame(columns=column_names, dtype="f8")

    if not contains_only_polygons(polygons):
        raise ValueError("Geoseries must contain only polygons.")

    # `cpp_polygon_bounding_boxes` computes all points supplied to the API,
    # but only supports single-polygon input. We compute the polygon offset
    # by combining the geometry offset and parts offset of the multipolygon
    # array.

    poly_offsets = polygons.polygons.part_offset.take(
        polygons.polygons.geometry_offset
    )
    ring_offsets = polygons.polygons.ring_offset
    x = polygons.polygons.x
    y = polygons.polygons.y

    return DataFrame._from_data(
        dict(
            zip(
                column_names,
                cpp_polygon_bounding_boxes(
                    as_column(poly_offsets),
                    as_column(ring_offsets),
                    as_column(x),
                    as_column(y),
                ),
            )
        )
    )


def linestring_bounding_boxes(linestring_offsets, xs, ys, expansion_radius):
    """Compute the minimum bounding boxes for a set of linestrings.

    Parameters
    ----------
    linestring_offsets
        Begin indices of the each linestring
    xs
        Linestring point x-coordinates
    ys
        Linestring point y-coordinates
    expansion_radius
        radius of each linestring point

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each linestring

        x_min : cudf.Series
            the minimum x-coordinate of each bounding box
        y_min : cudf.Series
            the minimum y-coordinate of each bounding box
        x_max : cudf.Series
            the maximum x-coordinate of each bounding box
        y_max : cudf.Series
            the maximum y-coordinate of each bounding box
    """
    linestring_offsets = as_column(linestring_offsets, dtype="int32")
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *cpp_linestring_bounding_boxes(
            linestring_offsets, xs, ys, expansion_radius
        )
    )
