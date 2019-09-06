# Copyright (c) 2019, NVIDIA CORPORATION.

from cudf import DataFrame
from cuspatial._lib.spatial import (
    cpp_directed_hausdorff_distance,
    cpp_haversine_distance,
    cpp_lonlat2coord,
    cpp_point_in_polygon_bitmap
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

    returns
    DataFrame: 'min', 'max' columns of Hausdorff distances for each polygon
    """
    return cpp_directed_hausdorff_distance(x, y, count)

def haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon):
    """ Compute the haversine distances between an arbitrary list of lat/lon
    pairs

    params
    p1_lat: latitude of first set of coords
    p1_lon: longitude of first set of coords
    p2_lat: latitude of second set of coords
    p2_lon: longitude of second set of coords
    
    Parameters
    ----------
    {params}

    returns
    Series: distance between all pairs of lat/lon coords
    """
    return cpp_haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon)

def lonlat_to_xy_km_coordinates(camera_lon, camera_lat, lon_coords, lat_coords):
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
    return DataFrame({'x': result[0],
                      'y': result[1]
    })

def point_in_polygon_bitmap(x_points, y_points,
        polygon_ids, polygon_end_indices, polygons_x, polygons_y):
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

    returns
    Series: one int32 for each point. This int32 is a binary bitmap specifying
    true or false for each of 32 polygons.
    """
    return cpp_point_in_polygon_bitmap(
        x_points, y_points,
        polygon_ids, polygon_end_indices, polygons_x, polygons_y
    )
