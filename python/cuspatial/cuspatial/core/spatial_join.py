# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib import spatial_join
from cuspatial.utils.column_utils import normalize_point_columns


def quad_bbox_join(quadtree, poly_bounding_boxes, x_min, x_max, y_min, y_max, scale, max_depth):
    """ Search a quadtree for polygon or polyline bounding box intersections.

    Parameters
    ----------
    quadtree : cudf.DataFrame
        A complete quadtree for a given area-of-interest bounding box.
    poly_bounding_boxes : cudf.DataFrame
        Minimum bounding boxes for a set of polygons or polylines
    x_min
        The lower-left x-coordinate of the area of interest bounding box
    x_max
        The upper-right x-coordinate of the area of interest bounding box
    min_y
        The lower-left y-coordinate of the area of interest bounding box
    max_y
        The upper-right y-coordinate of the area of interest bounding box
    scale
        Scale to apply to each point's distance from ``(x_min, y_min)``
    max_depth
        Maximum quadtree depth

    Returns
    -------
    result : cudf.DataFrame
        Indices for each intersecting bounding box and leaf quadrant.

        poly_offset : cudf.Series
            Indices for each poly bbox that intersects with the quadtree
        quad_offset : cudf.Series
            Indices for each leaf quadrant intersecting with a poly bbox

    Notes
    -----
    * Swaps ``min_x`` and ``max_x`` if ``min_x > max_x``
    * Swaps ``min_y`` and ``max_y`` if ``min_y > max_y``
    """
    return DataFrame._from_table(spatial_join.quad_bbox_join(
        quadtree, poly_bounding_boxes,
        x_min, x_max, y_min, y_max, scale, max_depth
    ))
