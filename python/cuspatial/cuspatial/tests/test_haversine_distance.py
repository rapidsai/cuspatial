# Copyright (c) 2019, NVIDIA CORPORATION.

"""
GPU-accelerated Haversine distance computation among three cities: New York, Paris and Sydney
Results match https://www.vcalc.com/wiki/vCalc/Haversine+-+Distance

Note: make sure cudf_dev conda environment is activated
"""

import pytest
import numpy as np
import cudf
from cudf.core import column
from cudf.tests.utils import assert_eq
import cuspatial.bindings.spatial as gis

def test_zeros():
    distance = gis.cpp_haversine_distance(
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
        cudf.Series([0.0]),
    )
    assert cudf.Series(distance)[0] == 0

def test_empty_x1():
    with pytest.raises(RuntimeError):
        distance = gis.cpp_haversine_distance(
            cudf.Series(),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0]),
        )

def test_empty_y1():
    with pytest.raises(RuntimeError):
        distance = gis.cpp_haversine_distance(
            cudf.Series([0]),
            cudf.Series(),
            cudf.Series([0]),
            cudf.Series([0]),
        )

def test_empty_x2():
    with pytest.raises(RuntimeError):
        distance = gis.cpp_haversine_distance(
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series(),
        )

def test_empty_y2():
    with pytest.raises(RuntimeError):
        distance = gis.cpp_haversine_distance(
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series([0]),
            cudf.Series(),
        )

def test_triple():
    cities = cudf.DataFrame({
        'New York': [-74.0060,40.7128],
        'Paris': [2.3522,48.8566],
        'Sydney': [151.2093,-33.8688]
    })
    cities = cities.set_index(['lat', 'lon'])
    pnt_x1 = []
    pnt_y1 = []
    pnt_x2 = []
    pnt_y2 = []
    for i in cities:
        for j in cities:
            pnt_x1.append(cities[i][0])
            pnt_y1.append(cities[i][1])
            pnt_x2.append(cities[j][0])
            pnt_y2.append(cities[j][1])
    distance = gis.cpp_haversine_distance(
        cudf.Series(pnt_x1),
        cudf.Series(pnt_y1),
        cudf.Series(pnt_x2),
        cudf.Series(pnt_y2)
    )
    assert np.allclose(distance.to_array(), [[0.0, 5.83724090e+03, 1.59887555e+04, 5.83724090e+03, 0.0, 1.69604974e+04, 1.59887555e+04, 1.69604974e+04, 0.0]])
