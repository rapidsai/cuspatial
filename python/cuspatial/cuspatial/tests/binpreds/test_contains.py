# Copyright (c) 2023, NVIDIA CORPORATION

import cupy as cp
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

import cuspatial

def test_same():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = lhs.contains(rhs)
    expected = gpdlhs.contains(gpdrhs)
    pd.testing.assert_series_equal(expected, got.to_pandas())

def test_adjacent():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])])
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = lhs.contains(rhs)
    expected = gpdlhs.contains(gpdrhs)
    pd.testing.assert_series_equal(expected, got.to_pandas())

def test_interior():
    lhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])])
    rhs = cuspatial.GeoSeries([Polygon([(0, 0), (0, 0.5), (0.5, 0.5), (0.5, 0)])])
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()
    got = lhs.contains(rhs)
    expected = gpdlhs.contains(gpdrhs)
    pd.testing.assert_series_equal(expected, got.to_pandas())
