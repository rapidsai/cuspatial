# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf

import cuspatial


def test_zeros():
    distance = cuspatial.haversine_distance(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
    )
    assert distance.element_indexing(0) == 0


def test_empty_x1():
    with pytest.raises(RuntimeError):
        distance = cuspatial.haversine_distance(  # noqa: F841
            cudf.Series(), cudf.Series([0]), cudf.Series([0]), cudf.Series([0])
        )


def test_empty_y1():
    with pytest.raises(RuntimeError):
        distance = cuspatial.haversine_distance(  # noqa: F841
            cudf.Series([0]), cudf.Series(), cudf.Series([0]), cudf.Series([0])
        )


def test_empty_x2():
    with pytest.raises(RuntimeError):
        distance = cuspatial.haversine_distance(  # noqa: F841
            cudf.Series([0]), cudf.Series([0]), cudf.Series([0]), cudf.Series()
        )


def test_empty_y2():
    with pytest.raises(RuntimeError):
        distance = cuspatial.haversine_distance(  # noqa: F841
            cudf.Series([0]), cudf.Series([0]), cudf.Series([0]), cudf.Series()
        )


def test_triple():
    cities = cudf.DataFrame(
        {
            "New York": [-74.0060, 40.7128],
            "Paris": [2.3522, 48.8566],
            "Sydney": [151.2093, -33.8688],
        }
    )
    cities.index = ["lat", "lon"]
    pnt_x1 = []
    pnt_y1 = []
    pnt_x2 = []
    pnt_y2 = []
    for i in cities:
        for j in cities:
            pnt_x1.append(cities[i].iloc[0])
            pnt_y1.append(cities[i].iloc[1])
            pnt_x2.append(cities[j].iloc[0])
            pnt_y2.append(cities[j].iloc[1])
    distance = cuspatial.haversine_distance(
        cudf.Series(pnt_x1),
        cudf.Series(pnt_y1),
        cudf.Series(pnt_x2),
        cudf.Series(pnt_y2),
    )
    assert np.allclose(
        distance.values_host,
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
