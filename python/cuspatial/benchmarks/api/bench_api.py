# Copyright (c) 2022, NVIDIA CORPORATION.

import cupy
import geopandas

import cudf

import cuspatial


def bench_io_read_polygon_shapefile(benchmark, shapefile):
    benchmark(cuspatial.read_polygon_shapefile, shapefile)


def bench_io_geoseries_from_offsets(benchmark, shapefile):
    shapefile_data = cuspatial.read_polygon_shapefile(shapefile)
    benchmark(cuspatial.core.geoseries.GeoSeries, shapefile_data)


def bench_io_from_geopandas(benchmark, host_dataframe):
    benchmark(cuspatial.from_geopandas, host_dataframe)


def bench_io_to_geopandas(benchmark, gpu_dataframe):
    benchmark(
        gpu_dataframe.to_geopandas,
    )


def bench_derive_trajectories(benchmark, sorted_trajectories):
    ids = cupy.random.randint(1, 400, 10000)
    timestamps = cupy.random.random(10000) * 10000
    x = cupy.random.random(10000)
    y = cupy.random.random(10000)
    benchmark(cuspatial.derive_trajectories, ids, x, y, timestamps)


def bench_trajectory_distances_and_speeds(benchmark, sorted_trajectories):
    length = len(cudf.Series(sorted_trajectories[1]).unique())
    benchmark(
        cuspatial.trajectory_distances_and_speeds,
        length,
        sorted_trajectories[0]["object_id"],
        sorted_trajectories[0]["x"],
        sorted_trajectories[0]["y"],
        sorted_trajectories[0]["timestamp"],
    )


def bench_trajectory_bounding_boxes(benchmark, sorted_trajectories):
    length = len(cudf.Series(sorted_trajectories[1]).unique())
    benchmark(
        cuspatial.trajectory_bounding_boxes,
        length,
        sorted_trajectories[0]["object_id"],
        sorted_trajectories[0]["x"],
        sorted_trajectories[0]["y"],
    )


def bench_polygon_bounding_boxes(benchmark, polygons):
    benchmark(
        cuspatial.polygon_bounding_boxes,
        polygons["geometry"].polygons.part_offset[:-1],
        polygons["geometry"].polygons.ring_offset[:-1],
        polygons["geometry"].polygons.x,
        polygons["geometry"].polygons.y,
    )


def bench_linestring_bounding_boxes(benchmark, sorted_trajectories):
    benchmark(
        cuspatial.linestring_bounding_boxes,
        sorted_trajectories[1],
        sorted_trajectories[0]["x"],
        sorted_trajectories[0]["y"],
        0.0001,
    )


def bench_sinusoidal_projection(benchmark, gpu_dataframe):
    afghanistan = gpu_dataframe["geometry"][
        gpu_dataframe["name"] == "Afghanistan"
    ]
    benchmark(
        cuspatial.sinusoidal_projection,
        afghanistan.polygons.y.mean(),
        afghanistan.polygons.x.mean(),
        afghanistan.polygons.y,
        afghanistan.polygons.x,
    )


def bench_directed_hausdorff_distance(benchmark, sorted_trajectories):
    benchmark(
        cuspatial.directed_hausdorff_distance,
        sorted_trajectories[0]["x"],
        sorted_trajectories[0]["y"],
        sorted_trajectories[1],
    )


def bench_haversine_distance(benchmark, gpu_dataframe):
    polygons_first = gpu_dataframe["geometry"][0:10]
    polygons_second = gpu_dataframe["geometry"][10:20]
    # The number of coordinates in two sets of polygons vary, so
    # we'll just compare the first set of 1000 values here.
    benchmark(
        cuspatial.haversine_distance,
        polygons_first.polygons.x[0:1000],
        polygons_first.polygons.y[0:1000],
        polygons_second.polygons.x[0:1000],
        polygons_second.polygons.y[0:1000],
    )


def bench_pairwise_linestring_distance(benchmark, gpu_dataframe):
    geometry = gpu_dataframe["geometry"]
    benchmark(
        cuspatial.pairwise_linestring_distance,
        geometry,
        geometry,
    )


def bench_points_in_spatial_window(benchmark, gpu_dataframe):
    geometry = gpu_dataframe["geometry"]
    mean_x, std_x = (geometry.polygons.x.mean(), geometry.polygons.x.std())
    mean_y, std_y = (geometry.polygons.y.mean(), geometry.polygons.y.std())
    benchmark(
        cuspatial.points_in_spatial_window,
        mean_x - std_x,
        mean_x + std_x,
        mean_y - std_y,
        mean_y + std_y,
        geometry.polygons.x,
        geometry.polygons.y,
    )


def bench_quadtree_on_points(benchmark, gpu_dataframe):
    polygons = gpu_dataframe["geometry"].polygons
    x_points = (cupy.random.random(10000000) - 0.5) * 360
    y_points = (cupy.random.random(10000000) - 0.5) * 180
    scale = 5
    max_depth = 7
    min_size = 125
    benchmark(
        cuspatial.quadtree_on_points,
        x_points,
        y_points,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth,
        min_size,
    )


def bench_quadtree_point_in_polygon(benchmark, polygons):
    polygons = polygons["geometry"].polygons
    x_points = (cupy.random.random(50000000) - 0.5) * 360
    y_points = (cupy.random.random(50000000) - 0.5) * 180
    scale = 5
    max_depth = 7
    min_size = 125
    point_indices, quadtree = cuspatial.quadtree_on_points(
        x_points,
        y_points,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth,
        min_size,
    )
    poly_bboxes = cuspatial.polygon_bounding_boxes(
        polygons.part_offset, polygons.ring_offset, polygons.x, polygons.y
    )
    intersections = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree,
        poly_bboxes,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth,
    )
    benchmark(
        cuspatial.quadtree_point_in_polygon,
        intersections,
        quadtree,
        point_indices,
        x_points,
        y_points,
        polygons.part_offset,
        polygons.ring_offset,
        polygons.x,
        polygons.y,
    )


def bench_quadtree_point_to_nearest_linestring(benchmark):
    SCALE = 3
    MAX_DEPTH = 7
    MIN_SIZE = 125
    host_countries = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres")
    )
    host_cities = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_cities")
    )
    gpu_countries = cuspatial.from_geopandas(
        host_countries[host_countries["geometry"].type == "Polygon"]
    )
    gpu_cities = cuspatial.from_geopandas(host_cities)
    polygons = gpu_countries["geometry"].polygons
    points_x = gpu_cities["geometry"].points.x
    points_y = gpu_cities["geometry"].points.y
    point_indices, quadtree = cuspatial.quadtree_on_points(
        points_x,
        points_y,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        SCALE,
        MAX_DEPTH,
        MIN_SIZE,
    )
    linestring_bboxes = cuspatial.linestring_bounding_boxes(
        polygons.ring_offset, polygons.x, polygons.y, 2.0
    )
    intersections = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree,
        linestring_bboxes,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        SCALE,
        MAX_DEPTH,
    )
    benchmark(
        cuspatial.quadtree_point_to_nearest_linestring,
        intersections,
        quadtree,
        point_indices,
        points_x,
        points_y,
        polygons.ring_offset,
        polygons.x,
        polygons.y,
    )


def bench_point_in_polygon(benchmark, gpu_dataframe):
    x_points = (cupy.random.random(50000000) - 0.5) * 360
    y_points = (cupy.random.random(50000000) - 0.5) * 180
    short_dataframe = gpu_dataframe.iloc[0:32]
    geometry = short_dataframe["geometry"]
    polygon_offset = cudf.Series(geometry.polygons.geometry_offset[0:31])
    benchmark(
        cuspatial.point_in_polygon,
        x_points,
        y_points,
        polygon_offset,
        geometry.polygons.ring_offset,
        geometry.polygons.x,
        geometry.polygons.y,
    )
