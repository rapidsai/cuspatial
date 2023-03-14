# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from math import ceil, sqrt

from cudf import DataFrame, Series
from cudf.core.column import NumericalColumn, as_column

import cuspatial
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_byte_point_in_polygon,
)
from cuspatial.utils.column_utils import normalize_point_columns
from cuspatial.utils.join_utils import pip_bitmap_column_to_binary_array

def contains_properly_quadtree(points, polygons):
    """Compute from a series of points and a series of polygons which points
    are properly contained within the corresponding polygon. Polygon A contains
    Point B properly if B intersects the interior of A but not the boundary (or
    exterior).

    Note that polygons must be closed: the first and last vertex of each
    polygon must be the same.

    Parameters
    ----------
    points : GeoSeries
        A GeoSeries of points.
    polygons : GeoSeries
        A GeoSeries of polygons.

    Returns
    -------
    result : cudf.Series
        A Series of boolean values indicating whether each point falls
        within its corresponding polygon.
    """

    scale = -1
    max_depth = 15
    min_size = ceil(sqrt(len(points)))
    if len(polygons) == 0:
        return Series()
    x_max = polygons.polygons.x.max()
    x_min = polygons.polygons.x.min()
    y_max = polygons.polygons.y.max()
    y_min = polygons.polygons.y.min()
    point_indices, quadtree = cuspatial.quadtree_on_points(
        points,
        x_min,
        x_max,
        y_min,
        y_max,
        scale,
        max_depth,
        min_size,
    )
    poly_bboxes = cuspatial.polygon_bounding_boxes(polygons)
    intersections = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree, poly_bboxes, x_min, x_max, y_min, y_max, scale, max_depth
    )
    polygons_and_points = cuspatial.quadtree_point_in_polygon(
        intersections, quadtree, point_indices, points, polygons
    )
    polygons_and_points["point_index"] = point_indices.iloc[
        polygons_and_points["point_index"]
    ].reset_index(drop=True)
    polygons_and_points["part_index"] = polygons_and_points["polygon_index"]
    polygons_and_points.drop("polygon_index", axis=1, inplace=True)
    return polygons_and_points


def contains_properly_byte_limited(points, polygons):
    """Compute from a series of points and a series of polygons which points
    are properly contained within the corresponding polygon. Polygon A contains
    Point B properly if B intersects the interior of A but not the boundary (or
    exterior).

    Note that polygons must be closed: the first and last vertex of each
    polygon must be the same.

    Parameters
    ----------
    points : GeoSeries
        A GeoSeries of points.
    polygons : GeoSeries
        A GeoSeries of polygons.

    Returns
    -------
    result : cudf.DataFrame
        A DataFrame of boolean values indicating whether each point falls
        within its corresponding polygon.
    """
    pip_result = cpp_byte_point_in_polygon(
        points.points.x,
        points.points.y,
        polygons.polygons.part_offset,
        polygons.polygons.ring_offset,
        polyons.polygons.x,
        polygons.polygons.y,
    )
    result = DataFrame(
        pip_bitmap_column_to_binary_array(
            polygon_bitmap_column=pip_result, width=len(poly_offsets) - 1
        )
    )
    final_result = DataFrame._from_data(
        {
            name: result[name].astype("bool")
            for name in reversed(result.columns)
        }
    )
    final_result.columns = range(len(final_result.columns))
    return final_result
