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
    """ Compute the directed Hausdorff distances between any groupings
    of polygons.

    params
    x: x coordinates
    y: y coordinates
    count: size of each polygon

    Parameters
    ----------
    {params}

    returns
    DataFrame: 'min', 'max' columns of Hausdorff distances for each polygon
    """
    return cpp_directed_hausdorff_distance(x, y, count)


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

    result_binary = gis_utils.pip_bitmap_column_to_boolean_array(bitmap_result)
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
