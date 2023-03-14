# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest

"""
# Tests
with contains_properly
31 polygons 31 points        - brute force pip
31 polyons 50000 points      - brute force pip
31 polygons 1_000_000 points - brute force pip
50_000 polygons 31 points           - columnar pip
50_000 polygons 50_000 points       - columnar pip
50_000 polygons 1_000_000 points    - columnar pip
1_000_000 polygons 31 points        - columnar pip
1_000_000 polygons 50000 points     - columnar pip
1_000_000 polygons 1_000_000 points - columnar pip
"""


@pytest.mark.benchmark(
    group="brute-force low polygon benchmarking",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize(
    "rhs_size",
    [31, 30_000, 1_000_000, 10_000_000, 20_000_000, 30_000_000, 69_000_000],
)
@pytest.mark.parametrize("lhs_size", [31])
def bench_contains_properly_low_poly(
    benchmark,
    lhs_size,
    rhs_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)][0]
    points = [*fast_point_generator(rhs_size)][0]
    benchmark(polygons.contains_properly, points, mode="pairs")


@pytest.mark.benchmark(
    group="quadtree low polygon benchmarking",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize(
    "rhs_size",
    [31, 30_000, 1_000_000, 10_000_000, 20_000_000, 30_000_000, 62_000_000],
)
@pytest.mark.parametrize("lhs_size", [31])
def bench_contains_properly_low_poly_allpairs(
    benchmark,
    lhs_size,
    rhs_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)][0]
    points = [*fast_point_generator(rhs_size)][0]
    benchmark(polygons.contains_properly, points, mode="allpairs")


@pytest.mark.benchmark(
    group="brute-force benchmarking",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize("rhs_size", [31, 30_000, 1_000_000])
@pytest.mark.parametrize("lhs_size", [31, 30_000, 1_000_000])
def bench_contains_properly(
    benchmark,
    lhs_size,
    rhs_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)][0]
    points = [*fast_point_generator(rhs_size)][0]
    benchmark(polygons.contains_properly, points, mode="pairs")


@pytest.mark.benchmark(
    group="quadtree",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize("rhs_size", [31, 30_000, 1_000_000, 10_000_000])
@pytest.mark.parametrize("lhs_size", [31, 30_000, 1_000_000, 10_000_000])
def bench_contains_properly_allpairs(
    benchmark,
    lhs_size,
    rhs_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)][0]
    points = [*fast_point_generator(rhs_size)][0]
    benchmark(polygons.contains_properly, points, mode="allpairs")


@pytest.mark.benchmark(
    group="pairwise",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize(
    "pair_size", [31, 310, 3_100, 31_000, 310_000, 3_100_000]
)
def bench_contains_properly_pairwise(
    benchmark,
    pair_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(pair_size, 0)][0]
    points = [*fast_point_generator(pair_size)][0]
    benchmark(polygons.contains_properly, points, mode="pairs")


@pytest.mark.benchmark(
    group="low polygon benchmarking, geopandas",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize("rhs_size", [31, 30_000, 1_000_000])
@pytest.mark.parametrize("lhs_size", [31])
def bench_contains_properly_low_poly_geopandas(
    benchmark,
    lhs_size,
    rhs_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)][0].to_geopandas()
    points = [*fast_point_generator(rhs_size)][0].to_geopandas()
    benchmark(polygons.contains, points)


@pytest.mark.benchmark(
    group="geopandas",
    min_time=0.1,
    max_time=0.5,
    min_rounds=1,
    disable_gc=False,
    warmup=False,
)
@pytest.mark.parametrize("rhs_size", [31, 30_000, 100_000])
@pytest.mark.parametrize("lhs_size", [31, 30_000, 100_000])
def bench_contains_properly_geopandas(
    benchmark,
    lhs_size,
    rhs_size,
    n_duplicated_polygons,
    fast_point_generator,
):
    polygons = [*n_duplicated_polygons(lhs_size, 0)][0].to_geopandas()
    points = [*fast_point_generator(rhs_size)][0].to_geopandas()
    benchmark(polygons.contains, points)
