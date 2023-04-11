# Copyright (c) 2022, NVIDIA CORPORATION.

import cupy
import geopandas

import cudf

import cuspatial


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
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": x, "y": y}).interleave_columns()
    )
    benchmark(cuspatial.derive_trajectories, ids, points, timestamps)


def bench_trajectory_distances_and_speeds(benchmark, sorted_trajectories):
    length = len(cudf.Series(sorted_trajectories[1]).unique())
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame(
            {
                "x": sorted_trajectories[0]["x"],
                "y": sorted_trajectories[0]["y"],
            }
        ).interleave_columns()
    )
    benchmark(
        cuspatial.trajectory_distances_and_speeds,
        length,
        sorted_trajectories[0]["object_id"],
        points,
        sorted_trajectories[0]["timestamp"],
    )


def bench_trajectory_bounding_boxes(benchmark, sorted_trajectories):
    length = len(cudf.Series(sorted_trajectories[1]).unique())
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame(
            {
                "x": sorted_trajectories[0]["x"],
                "y": sorted_trajectories[0]["y"],
            }
        ).interleave_columns()
    )
    benchmark(
        cuspatial.trajectory_bounding_boxes,
        length,
        sorted_trajectories[0]["object_id"],
        points,
    )


def bench_polygon_bounding_boxes(benchmark, polygons):
    benchmark(cuspatial.polygon_bounding_boxes, polygons)


def bench_linestring_bounding_boxes(benchmark, sorted_trajectories):
    xy = sorted_trajectories[0][["x", "y"]].interleave_columns()
    lines = cuspatial.GeoSeries.from_linestrings_xy(
        xy, sorted_trajectories[1], cupy.arange(len(sorted_trajectories))
    )
    benchmark(
        cuspatial.linestring_bounding_boxes,
        lines,
        0.0001,
    )


def bench_sinusoidal_projection(benchmark, gpu_dataframe):
    afghanistan = gpu_dataframe["geometry"][
        gpu_dataframe["name"] == "Afghanistan"
    ]
    lonlat = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame(
            {"lon": afghanistan.polygons.y, "lat": afghanistan.polygons.x}
        ).interleave_columns()
    )

    benchmark(
        cuspatial.sinusoidal_projection,
        afghanistan.polygons.y.mean(),
        afghanistan.polygons.x.mean(),
        lonlat,
    )


def bench_directed_hausdorff_distance(benchmark, sorted_trajectories):
    coords = sorted_trajectories[0][["x", "y"]].interleave_columns()
    offsets = sorted_trajectories[1]
    s = cuspatial.GeoSeries.from_multipoints_xy(coords, offsets)
    benchmark(cuspatial.directed_hausdorff_distance, s)


def bench_directed_hausdorff_distance_many_spaces(benchmark):
    spaces = 10000
    coords = cupy.zeros((spaces * 2,))
    offsets = cupy.arange(spaces + 1, dtype="int32")
    s = cuspatial.GeoSeries.from_multipoints_xy(coords, offsets)
    benchmark(cuspatial.directed_hausdorff_distance, s)


def bench_haversine_distance(benchmark, gpu_dataframe):
    coords_first = gpu_dataframe["geometry"][0:10].polygons.xy[0:1000]
    coords_second = gpu_dataframe["geometry"][10:20].polygons.xy[0:1000]

    points_first = cuspatial.GeoSeries.from_points_xy(coords_first)
    points_second = cuspatial.GeoSeries.from_points_xy(coords_second)

    benchmark(cuspatial.haversine_distance, points_first, points_second)


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
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame(
            {"x": geometry.polygons.x, "y": geometry.polygons.y}
        ).interleave_columns()
    )
    benchmark(
        cuspatial.points_in_spatial_window,
        points,
        mean_x - std_x,
        mean_x + std_x,
        mean_y - std_y,
        mean_y + std_y,
    )


def bench_quadtree_on_points(benchmark, gpu_dataframe):
    polygons = gpu_dataframe["geometry"].polygons
    x_points = (cupy.random.random(10000000) - 0.5) * 360
    y_points = (cupy.random.random(10000000) - 0.5) * 180
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": x_points, "y": y_points}).interleave_columns()
    )

    scale = 5
    max_depth = 7
    min_size = 125
    benchmark(
        cuspatial.quadtree_on_points,
        points,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth,
        min_size,
    )


def bench_quadtree_point_in_polygon(benchmark, polygons):
    df = polygons
    polygons = polygons["geometry"].polygons
    x_points = (cupy.random.random(50000000) - 0.5) * 360
    y_points = (cupy.random.random(50000000) - 0.5) * 180
    scale = 5
    max_depth = 7
    min_size = 125
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": x_points, "y": y_points}).interleave_columns()
    )
    point_indices, quadtree = cuspatial.quadtree_on_points(
        points,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        scale,
        max_depth,
        min_size,
    )
    poly_bboxes = cuspatial.polygon_bounding_boxes(df["geometry"])
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
        points,
        df["geometry"],
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
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": points_x, "y": points_y}).interleave_columns()
    )

    linestrings = cuspatial.GeoSeries.from_linestrings_xy(
        cudf.DataFrame(
            {"x": polygons.x, "y": polygons.y}
        ).interleave_columns(),
        polygons.ring_offset,
        cupy.arange(len(polygons.ring_offset)),
    )
    point_indices, quadtree = cuspatial.quadtree_on_points(
        points,
        polygons.x.min(),
        polygons.x.max(),
        polygons.y.min(),
        polygons.y.max(),
        SCALE,
        MAX_DEPTH,
        MIN_SIZE,
    )
    xy = cudf.DataFrame(
        {"x": polygons.x, "y": polygons.y}
    ).interleave_columns()
    lines = cuspatial.GeoSeries.from_linestrings_xy(
        xy, polygons.ring_offset, cupy.arange(len(polygons.ring_offset))
    )
    linestring_bboxes = cuspatial.linestring_bounding_boxes(lines, 2.0)
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
        points,
        linestrings,
    )


def bench_point_in_polygon(benchmark, polygons):
    x_points = (cupy.random.random(5000) - 0.5) * 360
    y_points = (cupy.random.random(5000) - 0.5) * 180
    points = cuspatial.GeoSeries.from_points_xy(
        cudf.DataFrame({"x": x_points, "y": y_points}).interleave_columns()
    )
    short_dataframe = polygons.iloc[0:31]
    geometry = short_dataframe["geometry"]
    benchmark(cuspatial.point_in_polygon, points, geometry)
