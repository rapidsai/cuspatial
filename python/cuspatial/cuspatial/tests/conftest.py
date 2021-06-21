# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


@pytest.fixture
def gs():
    g0 = Point(-1, 0)
    g1 = MultiPoint(((1, 2), (3, 4)))
    g2 = MultiPoint(((5, 6), (7, 8)))
    g3 = Point(9, 10)
    g4 = LineString(((11, 12), (13, 14)))
    g5 = MultiLineString((((15, 16), (17, 18)), ((19, 20), (21, 22))))
    g6 = MultiLineString((((23, 24), (25, 26)), ((27, 28), (29, 30))))
    g7 = LineString(((31, 32), (33, 34)))
    g8 = Polygon(((35, 36), (37, 38), (39, 40), (41, 42)),)
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


@pytest.fixture
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


@pytest.fixture
def polys():
    return np.array(
        (
            (35, 36),
            (37, 38),
            (39, 40),
            (41, 42),
            (35, 36),
            (43, 44),
            (45, 46),
            (47, 48),
            (43, 44),
            (49, 50),
            (51, 52),
            (53, 54),
            (49, 50),
            (55, 56),
            (57, 58),
            (59, 60),
            (55, 56),
            (61, 62),
            (63, 64),
            (65, 66),
            (61, 62),
            (67, 68),
            (69, 70),
            (71, 72),
            (67, 68),
            (73, 74),
            (75, 76),
            (77, 78),
            (73, 74),
            (79, 80),
            (81, 82),
            (83, 84),
            (79, 80),
            (85, 86),
            (87, 88),
            (89, 90),
            (85, 86),
            (91, 92),
            (93, 94),
            (95, 96),
            (91, 92),
            (97, 98),
            (99, 101),
            (102, 103),
            (101, 108),
            (97, 98),
            (106, 107),
            (108, 109),
            (110, 111),
            (113, 108),
            (106, 107),
        )
    )


@pytest.fixture
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
