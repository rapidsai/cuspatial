# Copyright (c) 2019-2023, NVIDIA CORPORATION.

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

import cuspatial


def test_zeros():
    distance = cuspatial.haversine_distance(
        cuspatial.GeoSeries([Point(0, 0)]), cuspatial.GeoSeries([Point(0, 0)])
    )
    assert np.allclose(distance.to_numpy(), [0.0])


def test_triple():
    cities = gpd.GeoSeries(
        [
            Point(-74.0060, 40.7128),
            Point(2.3522, 48.8566),
            Point(151.2093, -33.8688),
        ],
        index=["New York", "Paris", "Sydney"],
    )

    # Compute all pairs from pairwise
    cities1 = cuspatial.from_geopandas(cities.repeat(3))
    cities2 = cuspatial.from_geopandas(pd.concat([cities] * 3))

    distance = cuspatial.haversine_distance(
        cities1,
        cities2,
    )
    assert np.allclose(
        distance.to_numpy(),
        [
            [
                0.0,
                5.83724090e03,
                1.59887555e04,
                5.83724090e03,
                0.0,
                1.69604974e04,
                1.59887555e04,
                1.69604974e04,
                0.0,
            ]
        ],
    )
