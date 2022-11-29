# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf import Series
from cudf.core.column import as_column

from cuspatial._lib.pairwise_point_in_polygon import (
    pairwise_point_in_polygon as cpp_pairwise_point_in_polygon,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial.utils.column_utils import normalize_point_columns


def contains(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """Compute from a set of points and a set of polygons which points fall
    within each polygon. Note that `polygons_(x,y)` must be specified as
    closed polygons: the first and last coordinate of each polygon must be
    the same.

    This function is pairwise, or many-to-one.

    This implements `.contains_properly`, which shares a large
    space of correct cases with `GeoPandas.contains` but they do not produce
    identical results. In the future we will use intersection testing to
    match .contains behavior.

    Parameters
    ----------
    test_points_x
        x-coordinate of test points
    test_points_y
        y-coordinate of test points
    poly_offsets
        beginning index of the first ring in each polygon
    poly_ring_offsets
        beginning index of the first point in each ring
    poly_points_x
        x closed-coordinate of polygon points
    poly_points_y
        y closed-coordinate of polygon points

    Examples
    --------

    Test if a polygon is inside another polygon:

    >>> gpdpoint = gpd.GeoSeries(
        [Point(0.5, 0.5)],
        )
    >>> gpdpolygon = gpd.GeoSeries(
        [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
        ]
    )
    >>> point = cuspatial.from_geopandas(gpdpoint)
    >>> polygon = cuspatial.from_geopandas(gpdpolygon)
    >>> print(polygon.contains(point))
    0    False
    dtype: bool


    Test whether 3 points fall within either of two polygons

    >>> gpdpoint = gpd.GeoSeries(
        [Point(0, 0)],
        [Point(0, 0)],
        [Point(0, 0)],
        [Point(-1, 1)],
        [Point(-1, 1)],
        [Point(-1, 1)],
        )
    >>> gpdpolygon = gpd.GeoSeries(
        [
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[0, 0], [1, 0], [1, 1], [0, 0]]),
            Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
            Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
            Polygon([[-2, -2], [-2, 2], [2, 2], [-2, -2]]),
        ]
    )
    >>> point = cuspatial.from_geopandas(gpdpoint)
    >>> polygon = cuspatial.from_geopandas(gpdpolygon)
    >>> print(polygon.contains(point))
    0    False
    1    False
    2    False
    3     True
    4     True
    5     True
    dtype: bool

    note
    input Series x and y will not be index aligned, but computed as
    sequential arrays.

    note
    poly_ring_offsets must contain only the rings that make up the polygons
    indexed by poly_offsets. If there are rings in poly_ring_offsets that
    are not part of the polygons in poly_offsets, results are likely to be
    incorrect and behavior is undefined.

    Returns
    -------
    result : cudf.DataFrame
        A DataFrame of boolean values indicating whether each point falls
        within each polygon.
    """

    if len(poly_offsets) == 0:
        return Series()
    (
        test_points_x,
        test_points_y,
        poly_points_x,
        poly_points_y,
    ) = normalize_point_columns(
        as_column(test_points_x),
        as_column(test_points_y),
        as_column(poly_points_x),
        as_column(poly_points_y),
    )

    if len(test_points_x) == len(poly_offsets):
        pip_result = cpp_pairwise_point_in_polygon(
            test_points_x,
            test_points_y,
            as_column(poly_offsets, dtype="int32"),
            as_column(poly_ring_offsets, dtype="int32"),
            poly_points_x,
            poly_points_y,
        )
    else:
        pip_result = cpp_point_in_polygon(
            test_points_x,
            test_points_y,
            as_column(poly_offsets, dtype="int32"),
            as_column(poly_ring_offsets, dtype="int32"),
            poly_points_x,
            poly_points_y,
        )

    result = Series(pip_result, dtype="bool")
    return result
