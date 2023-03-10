# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest

"""
@pytest.mark.parametrize("size", [10, 50000, 1000000])
def bench_pairwise_contains_properly(
    benchmark,
    size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(size, 0)]
    points = [*fast_point_generator(size)]
    try:
        benchmark(polygons[0].contains_properly, points[0], mode="pairs")
    except Exception as e:
        print('OOM with lhs_size, rhs_size, and polygon_size:')
        print('        {lhs_size}, {rhs_size}, {polygon_size}')

@pytest.mark.parametrize("lhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("rhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("polygon_size", [4, 100, 50000])
def bench_quadtree_contains_properly(
    benchmark,
    lhs_size,
    rhs_size,
    polygon_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)]
    points = [*fast_point_generator(rhs_size)]
    try:
        benchmark(polygons[0].contains_properly, points[0], mode="allpairs")
    except e:
        print('OOM with lhs_size, rhs_size, and polygon_size:')
        print('        {lhs_size}, {rhs_size}, {polygon_size}')
"""


@pytest.mark.benchmark(
    group="pip benchmarking",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize("lhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("rhs_size", [10, 50000, 1000000])
@pytest.mark.parametrize("polygon_size", [4, 100, 50000])
def bench_column_length_limited_contains_properly(
    benchmark,
    lhs_size,
    rhs_size,
    polygon_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)]
    points = [*fast_point_generator(rhs_size)]
    benchmark(polygons[0].contains_properly, points[0], mode="pairs")


@pytest.mark.benchmark(
    group="pip benchmarking",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
def bench_column_limited_contains_properly(
    benchmark, n_duplicated_polygons, fast_point_generator
):
    polygons = [*n_duplicated_polygons(31, 0)]
    points = [*fast_point_generator(60000000)]
    benchmark(polygons[0].contains_properly, points[0], mode="pairs")
