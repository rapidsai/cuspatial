# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame

from cuspatial._lib.spatial import (
    cpp_directed_hausdorff_distance,
    cpp_haversine_distance,
    cpp_lonlat2coord,
    cpp_point_in_polygon_bitmap,
    cpp_spatial_window_points,
)
from cuspatial.utils import gis_utils


def directed_hausdorff_distance(x, y, count):
    """ Compute the directed Hausdorff distances between all pairs of
    trajectories.

    params
    x: x coordinates
    y: y coordinates
    count: size of each trajectory

    Parameters
    ----------
    {params}

    Example
    -------
    The directed Hausdorff distance from one trajectory to another is the
    greatest of all the distances from a point in the first trajectory to
    the closest point in the second.
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
    column per input trajectory; the value at row i, column j represents the
    hausdorff distance from trajectory i to trajectory j.
    """
    result = cpp_directed_hausdorff_distance(x, y, count)
    dim = len(count)
    return DataFrame.from_gpu_matrix(
        result.to_gpu_array().reshape(dim, dim)
    )


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
    return cpp_haversine_distance(p1_lon, p1_lat, p2_lon, p2_lat)


def lonlat_to_xy_km_coordinates(
    camera_lon, camera_lat, lon_coords, lat_coords
):
    """ Convert lonlat coordinates to km x,y coordinates based on some camera
    origin.

    params
    camera_lon: float64 - longitude camera
    camera_lat: float64 - latitude camera
    lon_coords: Series of longitude coords to convert to x
    lat_coords: Series of latitude coords to convert to y

    Parameters
    ----------
    {params}

    returns
    DataFrame: 'x', 'y' columns for new km positions of coords
    """
    result = cpp_lonlat2coord(camera_lon, camera_lat, lon_coords, lat_coords)
    return DataFrame({"x": result[0], "y": result[1]})


def point_in_polygon_bitmap(
    x_points,
    y_points,
    polygon_ids,
    polygon_end_indices,
    polygons_x,
    polygons_y,
):
    """ Compute from a set of points and a set of polygons which points fall
    within which polygons. Note that `polygons_(x,y)` must be specified as
    closed polygons: the first and last coordinate of each polygon must be
    the same.

    params
    x_points: x coordinates of points to test
    y_points: y coordinates of points to test
    polygon_ids: a unique integer id for each polygon
    polygon_end_indices: the (n+1)th vertex of the final coordinate of each
                         polygon in the next parameters
    polygons_x: x closed coordinates of all polygon points
    polygons_y: y closed coordinates of all polygon points

    Parameters
    ----------
    {params}

    Examples
    --------
        result = cuspatial.point_in_polygon_bitmap(
            cudf.Series([0, -8, 6.0]), # x coordinates of 3 query points
            cudf.Series([0, -8, 6.0]), # y coordinates of 3 query points
            cudf.Series([1, 2]), # unique id of two polygons
            cudf.Series([5, 10]), # position of last vertex in each polygon
            # polygon coordinates, x and y. Note [-10, -10] and [0, 0] repeat
            # the start/end coordinate of the two polygons.
            cudf.Series([-10.0, 5, 5, -10, -10, 0, 10, 10, 0, 0]),
            cudf.Series([-10.0, -10, 5, 5, -10, 0, 0, 10, 10, 0]),
        )
        # The result of point_in_polygon_bitmap is a DataFrame of Boolean
        # values indicating whether each point (rows) falls within
        # each polygon (columns).
        print(result)
                   in_polygon_1  in_polygon_2
        0          True          True
        1          True         False
        2         False          True

        # Point 0: (0, 0) falls in both polygons
        # Point 1: (-8, -8) falls in the first polygon
        # Point 2: (6.0, 6.0) falls in the second polygon

    returns
    DataFrame: a DataFrame of Boolean values indicating whether each point
    falls within each polygon.
    """
    bitmap_result = cpp_point_in_polygon_bitmap(
        x_points,
        y_points,
        polygon_ids,
        polygon_end_indices,
        polygons_x,
        polygons_y,
    )

    result_binary = gis_utils.pip_bitmap_column_to_binary_array(
        polygon_bitmap_column=bitmap_result, width=len(polygon_ids)
    )
    result_bools = DataFrame.from_gpu_matrix(
        result_binary
    )._apply_support_method("astype", dtype="bool")
    result_bools.columns = [
        f"in_polygon_{x}" for x in list(reversed(polygon_ids))
    ]
    result_bools = result_bools[list(reversed(result_bools.columns))]
    return result_bools


def window_points(left, bottom, right, top, x, y):
    """ Return only the subset of coordinates that fall within the numerically
    closed borders [,] of the defined bounding box.

    params
    left: x coordinate of window left boundary
    bottom: y coordinate of window bottom boundary
    right: x coordinate of window right boundary
    top: y coordinate of window top boundary
    x: Series of x coordinates that may fall within the window
    y: Series of y coordinates that may fall within the window

    Parameters
    ----------
    {params}

    Returns
    -------
    DataFrame: subset of x, y pairs above that fall within the window
    """
    result = cpp_spatial_window_points(left, bottom, right, top, x, y)
    return DataFrame({"x": result[0], "y": result[1]})
