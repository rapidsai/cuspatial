# Copyright (c) 2022 NVIDIA CORPORATION.

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import cuspatial


def test_align_more_values():
    gpdlhs = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
        ]
    )
    gpdrhs = gpdlhs.iloc[0:1]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdrhs.align(gpdlhs)
    got = rhs.align(lhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_some_different_values():
    gpdlhs = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdlhs = gpdlhs
    gpdlhs.index = [0, 1, 2, 6, 7, 8]
    gpdrhs = gpdlhs
    gpdrhs.index = [0, 1, 2, 3, 4, 5]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_more_and_some_different_values():
    gpdlhs = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdrhs = gpdlhs[0:3]
    gpdrhs.index = [0, 1, 2]
    gpdlhs = gpdlhs
    gpdlhs.index = [0, 1, 2, 3, 4, 5]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdrhs.align(gpdlhs)
    got = rhs.align(lhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_with_slice():
    gpdlhs = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    lhs = cuspatial.from_geopandas(gpdlhs)
    gpdlhs = gpdlhs[0:3]
    gpdrhs = gpdlhs[3:6]
    lhs = lhs[0:3]
    rhs = lhs[3:6]
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdrhs.align(gpdlhs)
    got = rhs.align(lhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_out_of_orders_values():
    gpdlhs = gpd.GeoSeries(
        [
            None,
            None,
            None,
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdrhs = gpdlhs.iloc[np.random.permutation(len(gpdlhs))]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdrhs.align(gpdlhs)
    got = rhs.align(lhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_same_index():
    gpdlhs = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdrhs = gpdlhs
    gpdlhs.index = [0, 1, 2, 3, 4, 5]
    gpdrhs.index = [0, 1, 2, 3, 4, 5]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdrhs.align(gpdlhs)
    got = rhs.align(lhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_similar_index():
    gpdlhs = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdrhs = gpdlhs
    gpdlhs.index = [0, 1, 2, 3, 4, 5]
    gpdrhs.index = [1, 2, 3, 4, 5, 6]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdrhs.align(gpdlhs)
    got = rhs.align(lhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
