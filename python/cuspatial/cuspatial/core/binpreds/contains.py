# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from math import ceil, sqrt

import cudf
from cudf import DataFrame, Series
from cudf.core.column import as_column

import cuspatial
from cuspatial._lib.pairwise_point_in_polygon import (
    pairwise_point_in_polygon as cpp_pairwise_point_in_polygon,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_byte_point_in_polygon,
)
from cuspatial.utils.join_utils import pip_bitmap_column_to_binary_array


def _quadtree_contains_properly(points, polygons):
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

    # Set the scale to the default minimum scale without triggering a warning.
    max_depth = 15
    min_size = ceil(sqrt(len(points)))
    if len(polygons) == 0:
        return Series()
    x_max = polygons.polygons.x.max()
    x_min = polygons.polygons.x.min()
    y_max = polygons.polygons.y.max()
    y_min = polygons.polygons.y.min()
    scale = max(x_max - x_min, y_max - y_min) / ((1 << max_depth) + 2)
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


def _brute_force_contains_properly(points, polygons):
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
        as_column(points.points.x),
        as_column(points.points.y),
        as_column(polygons.polygons.part_offset),
        as_column(polygons.polygons.ring_offset),
        as_column(polygons.polygons.x),
        as_column(polygons.polygons.y),
    )
    result = DataFrame(
        pip_bitmap_column_to_binary_array(
            polygon_bitmap_column=pip_result,
            width=len(polygons.polygons.part_offset) - 1,
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


def _pairwise_pip_result_to_point_polygon_index_pairs(pairwise_result):
    pairwise_result.columns = [
        "pairwise_index",
        "point_index",
        "result",
    ]
    result = pairwise_result[["point_index", "pairwise_index"]][
        pairwise_result["result"].astype("bool")
    ]
    result = result.sort_values(["point_index", "pairwise_index"]).reset_index(
        drop=True
    )
    return result


def _pairwise_contains_properly(points, polygons):
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
    pip_result = cpp_pairwise_point_in_polygon(
        as_column(points.points.x),
        as_column(points.points.y),
        as_column(polygons.polygons.part_offset),
        as_column(polygons.polygons.ring_offset),
        as_column(polygons.polygons.x),
        as_column(polygons.polygons.y),
    )
    # Pairwise returns a boolean column where the point and polygon index
    # always correspond. We can use this to create a dataframe with the
    # same shape as the quadtree result. Finally all the False results
    # are dropped, as quadtree doesn't report False results.
    quadtree_shaped_result = (
        cudf.Series(pip_result).reset_index().reset_index()
    )
    result = _pairwise_pip_result_to_point_polygon_index_pairs(
        quadtree_shaped_result
    )
    return result


def contains_properly(polygons, points, mode="pairwise"):
    if mode == "quadtree":
        return _quadtree_contains_properly(points, polygons)
    elif mode == "pairwise":
        return _pairwise_contains_properly(points, polygons)
    else:
        # Use stack to convert the result to the same shape as quadtree's
        # result, name the columns appropriately, and return the
        # two-column DataFrame.
        bitmask_result = _brute_force_contains_properly(points, polygons)
        quadtree_shaped_result = bitmask_result.stack().reset_index()
        result = _pairwise_pip_result_to_point_polygon_index_pairs(
            quadtree_shaped_result
        )
        result.columns = ["part_index", "point_index"]
        return result
