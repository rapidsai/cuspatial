import cuspatial
import shapefile
import time

import cupy as cp

from shapely.geometry import Point, Polygon


def cuspatial_haversine_distance(taxi_data):

    pickup_lon =  taxi_data['pickup_longitude']
    pickup_lat = taxi_data['pickup_latitude']
    dropoff_lon = taxi_data['dropoff_longitude']
    dropoff_lat = taxi_data['dropoff_latitude']
    cuspatial_dist = cuspatial.haversine_distance(pickup_lon,
                                                  pickup_lat,
                                                  dropoff_lon,
                                                  dropoff_lat)
    print("len of cuspatial_dist : ", len(cuspatial_dist))

    return cuspatial_dist

def cupy_haversine_distance(taxi_data):

    pickup_lon =  taxi_data['pickup_longitude'].to_array()
    pickup_lat = taxi_data['pickup_latitude'].to_array()
    dropoff_lon = taxi_data['dropoff_longitude'].to_array()
    dropoff_lat = taxi_data['dropoff_latitude'].to_array()
    cupy_distance = []
    for i in range(len(pickup_lon)):
    #cuspatial_dist = cuspatial.haversine_distance(pickup_lon,
        lon_sin = cp.square(cp.sin((pickup_lon[i]-dropoff_lon[i])/2))
        lat_sin = cp.square(cp.sin((pickup_lat[i]-dropoff_lat[i])/2))
        lon_cos = cp.cos(pickup_lon[i]) * cp.cos(dropoff_lon[i])
        haversine_sqrt = cp.sqrt(lon_sin + (lon_cos*lat_sin))
        cupy_distance.append(2*cp.arcsin(haversine_sqrt))
    
    print("len of cupy_distance : ", len(cupy_distance))
    return cupy_distance
