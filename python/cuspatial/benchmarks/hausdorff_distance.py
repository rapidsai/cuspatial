import cuspatial

from scipy.spatial.distance import directed_hausdorff


def cuspatial_hausdorff_distance(locust_data):

    x_coor =  locust_data['longitude'].to_array()
    y_coor = locust_data['latitude'].to_array()
    obj_id = locust_data['object_id'].to_array()
    time_info = locust_data['timestamp'].to_array()

    sorted_locust_data, offset_info = cuspatial.derive_trajectories(object_ids=obj_id,
                                                       xs=x_coor,
                                                       ys=y_coor,
                                                       timestamps=time_info)

    cuspatial_dist = cuspatial.directed_hausdorff_distance(xs=sorted_locust_data.x,
                                                           ys=sorted_locust_data.y,
                                                           space_offsets=offset_info)

    return cuspatial_dist


def scipy_hausdorff_distance(locust_data):

    x_coor =  locust_data['longitude'].to_array().reshape((-1,1))
    y_coor = locust_data['latitude'].to_array().reshape((-1,1))
    

    cupy_distance = directed_hausdorff(x_coor, y_coor)

    return cupy_distance
