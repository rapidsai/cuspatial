import cuspatial
import shapefile
import time

import cupy as cp

from shapely.geometry import Point, Polygon


def cuspatial_hausdorff_distance(locust_data):

    x_coor =  locust_data['longitude']
    y_coor = locust_data['latitude']
    obj_id = locust_data['object_id']
    time_info = locust_data['timestamp']

    sorted_locust_data, offset_info = cuspatial.derive_trajectories(object_ids=obj_id,
                                                       xs=x_coor,
                                                       ys=y_coor,
                                                       timestamps=time_info)

    cuspatial_dist = cuspatial.directed_hausdorff_distance(xs=x_coor,
                                                           ys=y_coor,
                                                           points_per_space=offset_info)

    return cuspatial_dist

def cupy_haversine_distance(taxi_data):

    pickup_lon =  taxi_data['pickup_longitude'].to_array()
    pickup_lat = taxi_data['pickup_latitude'].to_array()
    dropoff_lon = taxi_data['dropoff_longitude'].to_array()
    dropoff_lat = taxi_data['dropoff_latitude'].to_array()
    cupy_distance = []
    for i in range(len(pickup_lon)):
        lon_sin = cp.square(cp.sin((pickup_lon[i]-dropoff_lon[i])/2))
        lat_sin = cp.square(cp.sin((pickup_lat[i]-dropoff_lat[i])/2))
        lon_cos = cp.cos(pickup_lon[i]) * cp.cos(dropoff_lon[i])
        haversine_sqrt = cp.sqrt(lon_sin + (lon_cos*lat_sin))
        cupy_distance.append(2*cp.arcsin(haversine_sqrt))
    
    return cupy_distance
