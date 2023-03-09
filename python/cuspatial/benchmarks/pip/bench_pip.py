# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest

import cuspatial


@pytest.mark.parametrize("size", [10, 50000, 1000000])
def bench_pairwise_contains_properly(
    benchmark,
    size,
    polygon_generator,
    point_generator,
):
    polygons = cuspatial.GeoSeries([*polygon_generator(4, size)])
    points = cuspatial.GeoSeries([*point_generator(size)])
    benchmark(polygons.contains_properly, points, mode="pairs")


@pytest.mark.parametrize("lhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("rhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("polygon_size", [4, 100, 50000])
def bench_quadtree_contains_properly(
    benchmark,
    lhs_size,
    rhs_size,
    polygon_size,
    polygon_generator,
    point_generator,
):
    polygons = cuspatial.GeoSeries(
        [*polygon_generator(polygon_size, lhs_size)]
    )
    points = cuspatial.GeoSeries([*point_generator(rhs_size)])
    benchmark(polygons.contains_properly, points, mode="allpairs")


@pytest.mark.parametrize("lhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("rhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("polygon_size", [4, 100, 50000])
def bench_byte_limited_contains_properly(
    benchmark,
    lhs_size,
    rhs_size,
    polygon_size,
    polygon_generator,
    point_generator,
):
    polygons = cuspatial.GeoSeries(
        [*polygon_generator(polygon_size, lhs_size)]
    )
    points = cuspatial.GeoSeries([*point_generator(rhs_size)])
    benchmark(polygons.contains_properly, points, mode="pairs")


@pytest.mark.parametrize("lhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("rhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("polygon_size", [4, 100, 50000])
def bench_column_length_limited_contains_properly(
    benchmark,
    lhs_size,
    rhs_size,
    polygon_size,
    polygon_generator,
    point_generator,
):
    polygons = cuspatial.GeoSeries(
        [*polygon_generator(polygon_size, lhs_size)]
    )
    points = cuspatial.GeoSeries([*point_generator(rhs_size)])
    benchmark(polygons.contains_properly, points, mode="pairs")
