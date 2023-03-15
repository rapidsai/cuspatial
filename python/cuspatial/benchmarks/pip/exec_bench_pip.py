# Copyright 2023 NVIDIA Corporation

import os
import signal
import sys
import time
from enum import Enum
from functools import wraps

import numpy as np
from conftest import fast_point_generator, n_duplicated_polygons

import cudf

sys.path.append(os.path.abspath(os.path.join("..")))


def timeout(seconds):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            def handle_timeout(signum, frame):
                raise TimeoutError(
                    "Function execution exceeded the time limit"
                )

            # Set the alarm signal and the handler
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(seconds)

            try:
                result = function(*args, **kwargs)
            finally:
                # Cancel the alarm
                signal.alarm(0)

            return result

        return wrapper

    return decorator


class RESULT_TYPE(Enum):
    OOM = -1
    TIMEOUT = -2
    NOT_RUN = -3
    UNKNOWN = -4


couples = [
    [32, 31],
    [32, 30_000],
    [32, 1_000_000],
    [32, 20_000_000],
    [32, 69_000_000],
    [30_000, 31],
    [30_000, 30_000],
    [30_000, 1_000_000],
    [1_000_000, 31],
    [1_000_000, 30_000],
    [1_000_000, 1_000_000],
]

results_quadtree = {}
results_pairs = {}
results_geopandas = {}


@timeout(60)
def benchmark(fn, *args, **kwargs):
    start = time.time()
    fn(*args, **kwargs)
    return time.time() - start


def benchmark_handle_exceptions(fn, *args, **kwargs):
    try:
        return benchmark(fn, *args, **kwargs)
    except MemoryError as e:
        return RESULT_TYPE.OOM.value
    except TimeoutError as e:
        return RESULT_TYPE.TIMEOUT.value
    except Exception as e:
        return RESULT_TYPE.UNKNOWN.value


@timeout(60)
def device_to_host(*args):
    return [x.to_geopandas() for x in args]


if __name__ == "__main__":

    lhs_max = np.array(couples)[:, 0].max()
    rhs_max = np.array(couples)[:, 1].max()
    seed_polygons = [*n_duplicated_polygons(lhs_max, 0)][0]
    seed_points = [*fast_point_generator(rhs_max)][0]
    print("Copying data from host to device")
    start = time.time()
    # host_seed_polygons = seed_polygons.to_geopandas()
    # host_seed_points = seed_points.to_geopandas()
    print(f"Time to copy {lhs_max} polygons and {rhs_max} points:")
    print(f"{time.time() - start}")

    for couple in couples:
        test = "unknown"
        couple_key = ",".join([str(c) for c in couple])
        # polygons = [*n_duplicated_polygons(couple[0], 0)][0]
        # points = [*fast_point_generator(couple[1])][0]
        polygons = seed_polygons[0 : couple[0]]
        points = seed_points[0 : couple[1]]

        print(f"Benching {len(polygons)} polygons and {len(points)} points")
        print("Benching brute force")
        pairs_speed = benchmark(
            polygons.contains_properly, points, mode="pairs"
        )

        print("Benching quadtree")
        allpairs_speed = benchmark(
            polygons.contains_properly, points, mode="allpairs"
        )

        # (gpdpolygons, gpdpoints) = device_to_host(polygons, points)
        gpdpolygons = host_seed_polygons[0 : couple[0]]
        gpdpoint = host_seed_points[0 : couple[1]]
        print("Benching geopandas")
        # index
        # set spatial index
        # query index
        # query with contains
        geopandas_speed = benchmark(gpdpolygons.contains, gpdpoints)

        results_pairs[couple_key] = pairs_speed
        results_quadtree[couple_key] = allpairs_speed
        results_geopandas[couple_key] = geopandas_speed

    results = {
        "quadtree": results_quadtree.values(),
        "pairwise": results_pairs.values(),
        "geopandas": results_geopandas.values(),
    }
    try:
        results_df = cudf.DataFrame(
            results,
            index=results_byte.keys(),
        )
        results_df.to_csv("benchmarks.csv")
    except Exception as e:
        print("Don't lose the data")
        breakpoint()
    print("Wrote benchmarks.csv")
