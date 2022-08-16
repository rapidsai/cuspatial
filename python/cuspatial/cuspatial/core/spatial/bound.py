from cuspatial._lib.polygon_bounding_boxes import (
    polygon_bounding_boxes as cpp_polygon_bounding_boxes,
)
from cuspatial._lib.polyline_bounding_boxes import (
    polyline_bounding_boxes as cpp_polyline_bounding_boxes,
)

from cuspatial.utils.column_utils import normalize_point_columns

from cudf.core.column import as_column
from cudf import DataFrame

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


def polyline_bounding_boxes(poly_offsets, xs, ys, expansion_radius):
    """Compute the minimum bounding-boxes for a set of polylines.

    Parameters
    ----------
    poly_offsets
        Begin indices of the first ring in each polyline (i.e. prefix-sum)
    xs
        Polyline point x-coordinates
    ys
        Polyline point y-coordinates
    expansion_radius
        radius of each polyline point

    Returns
    -------
    result : cudf.DataFrame
        minimum bounding boxes for each polyline

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
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    return DataFrame._from_data(
        *cpp_polyline_bounding_boxes(poly_offsets, xs, ys, expansion_radius)
    )
