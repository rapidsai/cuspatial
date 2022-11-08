# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib.linestring_bounding_boxes import (
    linestring_bounding_boxes as cpp_linestring_bounding_boxes,
)
from cuspatial._lib.polygon_bounding_boxes import (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)
from cuspatial.utils.column_utils import normalize_point_columns


def polygon_bounding_boxes(poly_offsets, ring_offsets, xs, ys):
    """Compute the minimum bounding-boxes for a set of polygons.

    Parameters
    ----------
    poly_offsets
        Begin indices of the first ring in each polygon (i.e. prefix-sum)
    ring_offsets
        Begin indices of the first point in each ring (i.e. prefix-sum)
    xs
        Polygon point x-coordinates
    ys
        Polygon point y-coordinates

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each polygon

        x_min : cudf.Series
            the minimum x-coordinate of each bounding box
        y_min : cudf.Series
            the minimum y-coordinate of each bounding box
        x_max : cudf.Series
            the maximum x-coordinate of each bounding box
        y_max : cudf.Series
            the maximum y-coordinate of each bounding box
    """
    poly_offsets = as_column(poly_offsets, dtype="int32")
    ring_offsets = as_column(ring_offsets, dtype="int32")
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *cpp_polygon_bounding_boxes(poly_offsets, ring_offsets, xs, ys)
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
