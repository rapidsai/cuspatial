import cudf
import cuspatial
import shapefile
import time

from shapely.geometry import Point, Polygon


def points_in_polygon(taxi_data, taxi_zones, pickup=True):

    if pickup:
        p_x = taxi_data['pickup_longitude']
        p_y = taxi_data['pickup_latitude']
        poly_x = taxi_zones[2]['x']
        poly_y = taxi_zones[2]['y']
        x_min = p_x.min()
        x_max = p_x.max()
        y_min = p_y.min()
        y_max = p_y.max()
        scale = 1
        max_depth = 3
        min_size = 12
        point_indices, quadtree = cuspatial.quadtree_on_points(
            p_x,
            p_y,
            p_x.min(),
            p_x.max(),
            p_y.min(),
            p_y.max(),
            scale,
            max_depth,
            min_size,
        )
        expansion_radius = 2.0
        poly_bboxes = cuspatial.polyline_bounding_boxes(
            cudf.Series([0, 3, 8, 12]),
            poly_x,
            poly_y,
            expansion_radius,
        )
        intersections = cuspatial.join_quadtree_and_bounding_boxes(
            quadtree, poly_bboxes, x_min, x_max, y_min, y_max, scale, max_depth
        )
        polygons_and_points = cuspatial.quadtree_point_in_polygon(
            intersections,
            quadtree,
            point_indices,
            p_x,
            p_y,
            cudf.Series([1, 2, 3, 4]),
            cudf.Series([0, 3, 8, 12]),
            poly_x,
            poly_y
        )

        return polygons_and_points

    else:
        p_x = taxi_data['dropoff_longitude']
        p_y = taxi_data['dropoff_latitude']
        poly_x = taxi_zones[2]['x']
        poly_y = taxi_zones[2]['y']
        x_min = p_x.min()
        x_max = p_x.max()
        y_min = p_y.min()
        y_max = p_y.max()
        scale = 1
        max_depth = 3
        min_size = 12
        point_indices, quadtree = cuspatial.quadtree_point_in_polygon(
            p_x,
            p_y,
            p_x.min(),
            p_x.max(),
            p_y.min(),
            p_y.max(),
            scale,
            max_depth,
            min_size,
        )
        expansion_radius = 2.0
        poly_bboxes = cuspatial.polyline_bounding_boxes(
            cudf.Series([1, 2, 3, 4]),
            cudf.Series([0, 3, 8, 12]),
            poly_x,
            poly_y,
        )
        intersections = cuspatial.join_quadtree_and_bounding_boxes(
            quadtree, poly_bboxes, x_min, x_max, y_min, y_max, scale, max_depth
        )
        polygons_and_points = cuspatial.quadtree_point_in_polygon(
            intersections,
            quadtree,
            point_indices,
            p_x,
            p_y,
            cudf.Series([1, 2, 3, 4]),
            cudf.Series([0, 3, 8, 12]),
            poly_x,
            poly_y
        )

        return polygons_and_points


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
    print("end-start for cpu : ", end - start)
    return check_vals
