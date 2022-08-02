# Copyright (c) 2022, NVIDIA CORPORATION.

"""Benchmarks of GeoSeries methods."""

from config import cuspatial


def bench_from_geoseries_100(benchmark, gpdf_100):
    benchmark(cuspatial.from_geopandas, gpdf_100["geometry"])


def bench_from_geoseries_1000(benchmark, gpdf_1000):
    benchmark(cuspatial.from_geopandas, gpdf_1000["geometry"])


def bench_from_geoseries_10000(benchmark, gpdf_10000):
    benchmark(cuspatial.from_geopandas, gpdf_10000["geometry"])


def bench_from_geoseries_100000(benchmark, gpdf_100000):
    benchmark(cuspatial.from_geopandas, gpdf_100000["geometry"])
