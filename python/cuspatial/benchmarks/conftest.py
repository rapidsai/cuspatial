# Copyright (c) 2022, NVIDIA CORPORATION.

"""Defines pytest fixtures for all benchmarks.

The cuspatial fixture is a single randomly generated GeoDataframe, containing
4 columns: 1 GeoSeries column, an int column, a float column, and a string
column.
"""

import cupy as cp
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest_cases
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cuspatial


@pytest_cases.fixture
def gs():
    g0 = Point(-1, 0)
    g1 = MultiPoint(((1, 2), (3, 4)))
    g2 = MultiPoint(((5, 6), (7, 8)))
    g3 = Point(9, 10)
    g4 = LineString(((11, 12), (13, 14)))
    g5 = MultiLineString((((15, 16), (17, 18)), ((19, 20), (21, 22))))
    g6 = MultiLineString((((23, 24), (25, 26)), ((27, 28), (29, 30))))
    g7 = LineString(((31, 32), (33, 34)))
    g8 = Polygon(
        ((35, 36), (37, 38), (39, 40), (41, 42)),
    )
    g9 = MultiPolygon(
        [
            (
                ((43, 44), (45, 46), (47, 48)),
                [((49, 50), (51, 52), (53, 54))],
            ),
            (
                ((55, 56), (57, 58), (59, 60)),
                [((61, 62), (63, 64), (65, 66))],
            ),
        ]
    )
    g10 = MultiPolygon(
        [
            (
                ((67, 68), (69, 70), (71, 72)),
                [((73, 74), (75, 76), (77, 78))],
            ),
            (
                ((79, 80), (81, 82), (83, 84)),
                [
                    ((85, 86), (87, 88), (89, 90)),
                    ((91, 92), (93, 94), (95, 96)),
                ],
            ),
        ]
    )
    g11 = Polygon(
        ((97, 98), (99, 101), (102, 103), (101, 108)),
        [((106, 107), (108, 109), (110, 111), (113, 108))],
    )
    gs = gpd.GeoSeries([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])
    return gs


@pytest_cases.fixture
def gpdf(gs):
    int_col = list(range(len(gs)))
    random_col = int_col
    np.random.shuffle(random_col)
    str_col = [str(x) for x in int_col]
    key_col = np.repeat(np.arange(4), len(int_col) // 4)
    np.random.shuffle(key_col)
    result = gpd.GeoDataFrame(
        {
            "geometry": gs,
            "integer": int_col,
            "string": str_col,
            "random": random_col,
            "key": key_col,
        }
    )
    result["float"] = result["integer"].astype("float64")
    return result


@pytest_cases.fixture
def gs_sorted(gs):
    result = pd.concat(
        [
            gs[gs.type == "Point"],
            gs[gs.type == "MultiPoint"],
            gs[gs.type == "LineString"],
            gs[gs.type == "MultiLineString"],
            gs[gs.type == "Polygon"],
            gs[gs.type == "MultiPolygon"],
        ]
    )
    return result.reset_index(drop=True)


def make_geopandas_dataframe_from_naturalearth_lowres(nr):
    source_df = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    result_df = gpd.GeoDataFrame()
    for i in range((nr // len(source_df)) + 1):
        scramble_df = source_df.iloc[
            np.random.choice(len(source_df), len(source_df)), :
        ]
        result_df = pd.concat([scramble_df, result_df]).reset_index(drop=True)
    return result_df


@pytest_cases.fixture()
def host_dataframe():
    return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


@pytest_cases.fixture()
def gpu_dataframe(host_dataframe):
    return cuspatial.from_geopandas(host_dataframe)


@pytest_cases.fixture()
def polygons(host_dataframe):
    return cuspatial.from_geopandas(
        host_dataframe[host_dataframe["geometry"].type == "Polygon"]
    )


@pytest_cases.fixture()
def sorted_trajectories():
    ids = cp.random.randint(1, 400, 10000)
    timestamps = cp.random.random(10000) * 10000
    x = cp.random.random(10000)
    y = cp.random.random(10000)
    return cuspatial.derive_trajectories(ids, x, y, timestamps)


@pytest_cases.fixture()
def gpdf_100(request):
    return make_geopandas_dataframe_from_naturalearth_lowres(100)


@pytest_cases.fixture()
def gpdf_1000(request):
    return make_geopandas_dataframe_from_naturalearth_lowres(1000)


@pytest_cases.fixture()
def gpdf_10000(request):
    return make_geopandas_dataframe_from_naturalearth_lowres(10000)


@pytest_cases.fixture()
def cugpdf_100(gpdf_100):
    return cuspatial.from_geopandas(gpdf_100)


@pytest_cases.fixture()
def shapefile(tmp_path, gpdf_100):
    d = tmp_path / "shp"
    d.mkdir()
    p = d / "read_polygon_shapefile"
    gpdf_100.to_file(p)
    return p
