import cuspatial
import shapefile
import time

from shapely.geometry import Point, Polygon


def points_in_polygon(taxi_data, taxi_zones, pickup=True):
    """
    tzones = gpd.GeoDataFrame.from_file(polygon_shape_file)
    tzones = tzones[0:27]
    print(" tzones shape : ", tzones.shape)
    tzones.to_file('cu_taxi_zones.shp')
    """
    # polygon_shape_file = data_dir  + 'its_4326_roi.shp'

    if pickup:
        pickups = cuspatial.point_in_polygon(taxi_data['pickup_longitude'],
                                             taxi_data['pickup_latitude'],
                                             taxi_zones[0], taxi_zones[1],
                                             taxi_zones[2]['x'],
                                             taxi_zones[2]['y'])
   
        return pickups

    else:
        dropoffs = cuspatial.point_in_polygon(taxi_data['dropoff_longitude'],
                                              taxi_data['dropoff_latitude'],
                                              taxi_zones[0],
                                              taxi_zones[1],
                                              taxi_zones[2]['x'],
                                              taxi_zones[2]['y'])

        return dropoffs


def cpu_points_in_polygon(taxi_data, polygon_shape_file, pickup=True):
    if pickup:
        pnt_lon = taxi_data['pickup_longitude']
        pnt_lat = taxi_data['pickup_latitude']
    else:
        pnt_lon = taxi_data['dropoff_longitude']
        pnt_lat = taxi_data['dropoff_latitude']

    pntx = pnt_lon.to_array()
    pnty = pnt_lat.to_array()

    # polygon_shape_file = data_dir + "its_4326_roi.shp"
    plyreader = shapefile.Reader(polygon_shape_file)
    polygon = plyreader.shapes()
    plys = []
    for shape in polygon:
        plys.append(Polygon(shape.points))
    print(" plys shape : ", len(plys))
    check_vals = []
    check_per_poly = []
    print("pnt_lon.size : ", pnt_lon.size)
    start = time.time()
    for i in range(pnt_lon.size):
        pt = Point(pntx[i], pnty[i])
        res = 0
        for j in range(len(plys)):
            pip = plys[len(plys) - 1 - j].contains(pt)
            if pip:
                res |= 0x01 << (len(plys) - 1 - j)
                print(res)
            check_per_poly.append(res)
        check_vals.append(check_per_poly)
    end = time.time()
    print("end-start for cpu : ", end-start)
    return check_vals
