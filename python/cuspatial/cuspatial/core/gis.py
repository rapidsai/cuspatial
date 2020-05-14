# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf import DataFrame
from cudf.core.column import as_column

from cuspatial._lib.hausdorff import (
    directed_hausdorff_distance as cpp_directed_hausdorff_distance,
)
from cuspatial._lib.point_in_polygon import (
    point_in_polygon as cpp_point_in_polygon,
)
from cuspatial._lib.spatial import (
    haversine_distance as cpp_haversine_distance,
    lonlat_to_cartesian as cpp_lonlat_to_cartesian,
)
from cuspatial.utils import gis_utils
from cuspatial.utils.column_utils import normalize_point_columns


def directed_hausdorff_distance(xs, ys, points_per_space):
    """Compute the directed Hausdorff distances between all pairs of
    spaces.

    params
    xs: x-coordinates
    ys: y-coordinates
    points_per_space: number of points in each space

    Parameters
    ----------
    {params}

    Example
    -------
    The directed Hausdorff distance from one space to another is the greatest
    of all the distances between any point in the first space to the closest
    point in the second.

    [Wikipedia](https://en.wikipedia.org/wiki/Hausdorff_distance)

    Consider a pair of lines on a grid.

     |
     o
    -oxx-
     |
    o = [[0, 0], [0, 1]]
    x = [[1, 0], [2, 0]]

    o[0] is the closest point in o to x. The distance from o[0] to the farthest
    point in x = 2.
    x[0] is the closest point in x to o. The distance from x[0] to the farthest
    point in o = 1.414.

        result = cuspatial.directed_hausdorff_distance(
            cudf.Series([0, 1, 0, 0]),
            cudf.Series([0, 0, 1, 2]),
            cudf.Series([2, 2,]),
        )
        print(result)
             0         1
        0  0.0  1.414214
        1  2.0  0.000000

    Returns
    -------
    DataFrame: The pairwise directed distance matrix with one row and one
    column per input space; the value at row i, column j represents the
    hausdorff distance from space i to space j.
    """
    num_spaces = len(points_per_space)
    if num_spaces == 0:
        return DataFrame()
    xs, ys = normalize_point_columns(as_column(xs), as_column(ys))
    result = cpp_directed_hausdorff_distance(
        xs, ys, as_column(points_per_space, dtype="int32"),
    )
    result = result.data_array_view
    result = result.reshape(num_spaces, num_spaces)
    return DataFrame.from_gpu_matrix(result)


def haversine_distance(p1_lon, p1_lat, p2_lon, p2_lat):
    """ Compute the haversine distances between an arbitrary list of lon/lat
    pairs

    params
    p1_lon: longitude of first set of coords
    p1_lat: latitude of first set of coords
    p2_lon: longitude of second set of coords
    p2_lat: latitude of second set of coords

    Parameters
    ----------
    {params}

    returns
    Series: distance between all pairs of lat/lon coords
    """
    p1_lon, p1_lat, p2_lon, p2_lat = normalize_point_columns(
        as_column(p1_lon),
        as_column(p1_lat),
        as_column(p2_lon),
        as_column(p2_lat),
    )
    return cpp_haversine_distance(p1_lon, p1_lat, p2_lon, p2_lat)


def lonlat_to_cartesian(origin_lon, origin_lat, input_lon, input_lat):
    """ Convert lonlat coordinates to km x,y coordinates based on some camera
    origin.

    params
    origin_lon: float64 - longitude camera
    origin_lat: float64 - latitude camera
    input_lon: Series of longitude coords to convert to x
    input_lat: Series of latitude coords to convert to y

    Parameters
    ----------
    {params}

    returns
    DataFrame: 'x', 'y' columns for new km positions of coords
    """
    result = cpp_lonlat_to_cartesian(
        origin_lon, origin_lat, input_lon._column, input_lat._column
    )
    return DataFrame({"x": result[0], "y": result[1]})


def point_in_polygon(
    test_points_x,
    test_points_y,
    poly_offsets,
    poly_ring_offsets,
    poly_points_x,
    poly_points_y,
):
    """ Compute from a set of points and a set of polygons which points fall
    within which polygons. Note that `polygons_(x,y)` must be specified as
    closed polygons: the first and last coordinate of each polygon must be
    the same.

    params
    test_points_x: x-coordinate of test points
    test_points_y: y-coordinate of test points
    poly_offsets: beginning index of the first ring in each polygon
    poly_ring_offsets: beginning index of the first point in each ring
    poly_points_x: x closed-coordinate of polygon points
    poly_points_y: y closed-coordinate of polygon points

    Parameters
    ----------
    {params}

    Examples
    --------
        result = cuspatial.point_in_polygon(
            cudf.Series([0, -8, 6.0]]), # x coordinates of 3 query points
            cudf.Series([0, -8, 6.0]), # y coordinates of 3 query points
            cudf.Series([1, 2], index=['nyc', 'dc']), # ring positions of
                    # two polygons each with one ring
            cudf.Series([4, 8]), # positions of last vertex in each ring
            # polygon coordinates, x and y. Note [-10, -10] and [0, 0] repeat
            # the start/end coordinate of the two polygons.
            cudf.Series([-10, 5, 5, -10, -10, 0, 10, 10, 0, 0]),
            cudf.Series([-10, -10, 5, 5, -10, 0, 0, 10, 10, 0]),
        )
        # The result of point_in_polygon is a DataFrame of Boolean
        # values indicating whether each point (rows) falls within
        # each polygon (columns).
        print(result)
                    nyc            dc
        0          True          True
        1          True         False
        2         False          True

        # Point 0: (0, 0) falls in both polygons
        # Point 1: (-8, -8) falls in the first polygon
        # Point 2: (6.0, 6.0) falls in the second polygon

    note
    input Series x and y will not be index aligned, but computed as
    sequential arrays.

    returns
    DataFrame: a DataFrame of Boolean values indicating whether each point
    falls within each polygon.
    """

    if len(poly_offsets) == 0:
        return DataFrame()

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

    result = cpp_point_in_polygon(
        test_points_x,
        test_points_y,
        as_column(poly_offsets, dtype="int32"),
        as_column(poly_ring_offsets, dtype="int32"),
        poly_points_x,
        poly_points_y,
    )

    result = gis_utils.pip_bitmap_column_to_binary_array(
        polygon_bitmap_column=result, width=len(poly_offsets)
    )
    result = DataFrame.from_gpu_matrix(result)
    result = result._apply_support_method("astype", dtype="bool")
    result.columns = [x for x in list(reversed(poly_offsets.index))]
    result = result[list(reversed(result.columns))]
    return result
