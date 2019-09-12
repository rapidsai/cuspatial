# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame

from cuspatial._lib.spatial import (
    cpp_directed_hausdorff_distance,
    cpp_haversine_distance,
    cpp_lonlat2coord,
    cpp_point_in_polygon_bitmap,
    cpp_spatial_window_points,
)


def directed_hausdorff_distance(x, y, count):
    """ Compute the directed Hausdorff distances between any groupings
    of polygons.

    params
    x: x coordinates
    y: y coordinates
    count: size of each polygon

    Parameters
    ----------
    {params}

    Example
    -------
    Consider a pair of lines on a grid.

     |
     o
    -oxx-
     |
    o = [[0, 0], [0, 1]]
    x = [[1, 0], [2, 0]]

    o[0] is the nearer point in o to x. The distance from o[0] to the farthest
    point in x = 2.
    x[0] is the nearer point in x to o. The distance from x[0] to the farthest
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

    returns
    DataFrame: The pairwise hausdorff distance of each set to each other set.
    """
    result = cpp_directed_hausdorff_distance(x, y, count)
    dim = len(count)
    return DataFrame.from_gpu_matrix(
        result.data.to_gpu_array().reshape(dim, dim)
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
    within which polygons.

    params
    x_points: x coordinates of points to test
    y_points: y coordinates of points to test
    polygon_ids: a unique integer id for each polygon
    polygon_end_indices: the (n+1)th vertex of the final coordinate of each
                         polygon in the next parameters
    polygons_x: x coordinates of all polygon points
    polygons_y: y coordinates of all polygon points

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
            # polygon coordinates, x and y
            cudf.Series([-10.0, 5, 5, -10, -10, 0, 10, 10, 0, 0]),
            cudf.Series([-10.0, -10, 5, 5, -10, 0, 0, 10, 10, 0]),
        )
        # The result of point_in_polygon_bitmap is a binary bitmap of
        # coordinates inside of the polgyon.
        print(cudf.Series(result))
        0    3
        1    1
        2    2
        dtype: int32
        # The result 3, 1, 2 represents the position of each point in each
        # polygon in integer binary format:
        # Point 0: (0, 0) falls in both polygons: 0b11 (3)
        # Point 1: (-8, -8) falls in the first polygon: 0b01 (1)
        # Point 2: (6.0, 6.0) falls in the second polygon: 0b10 (2)

    returns
    Series: one int32 for each point. This int32 is a binary bitmap specifying
    true or false for each of 32 polygons.
    """
    return cpp_point_in_polygon_bitmap(
        x_points,
        y_points,
        polygon_ids,
        polygon_end_indices,
        polygons_x,
        polygons_y,
    )


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
