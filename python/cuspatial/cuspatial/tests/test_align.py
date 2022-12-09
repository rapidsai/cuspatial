# Copyright (c) 2022 NVIDIA CORPORATION.

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import cuspatial


def test_align_more_values():
    gpdpdf = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
        ]
    )
    gpdshort = gpdpdf.iloc[0:1]
    pdf = cuspatial.from_geopandas(gpdpdf)
    short = cuspatial.from_geopandas(gpdshort)
    expected = gpdshort.align(gpdpdf)
    got = short.align(pdf)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdpdf.align(gpdshort)
    got = pdf.align(short)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_some_different_values():
    gpdalign = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdlhs = gpdalign
    gpdalign.index = [0, 1, 2, 6, 7, 8]
    gpdrhs = gpdalign
    gpdrhs.index = [0, 1, 2, 3, 4, 5]
    lhs = cuspatial.from_geopandas(gpdlhs)
    rhs = cuspatial.from_geopandas(gpdrhs)
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_more_and_some_different_values():
    gpdalign = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    gpdrhs = gpdalign[0:3]
    gpdrhs.index = [0, 1, 2]
    gpdlhs = gpdalign
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
    gpdalign = gpd.GeoSeries(
        [
            Polygon(((1, 2), (3, 4), (5, 6), (7, 8))),
            Polygon(((9, 10), (11, 12), (13, 14), (15, 16))),
            Polygon(((17, 18), (19, 20), (21, 22), (23, 24))),
            Polygon(((25, 26), (27, 28), (29, 30), (31, 32))),
            Polygon(((33, 34), (35, 36), (37, 38), (39, 40))),
            Polygon(((41, 42), (43, 44), (45, 46), (47, 48))),
        ]
    )
    align = cuspatial.from_geopandas(gpdalign)
    gpdlhs = gpdalign[0:3]
    gpdrhs = gpdalign[3:6]
    lhs = align[0:3]
    rhs = align[3:6]
    expected = gpdlhs.align(gpdrhs)
    got = lhs.align(rhs)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())


def test_align_out_of_orders_values():
    gpdalign = gpd.GeoSeries(
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
    gpdreordered = gpdalign.iloc[np.random.permutation(len(gpdalign))]
    align = cuspatial.from_geopandas(gpdalign)
    reordered = cuspatial.from_geopandas(gpdreordered)
    expected = gpdalign.align(gpdreordered)
    got = align.align(reordered)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
    expected = gpdreordered.align(gpdalign)
    got = reordered.align(align)
    pd.testing.assert_series_equal(expected[0], got[0].to_pandas())
    pd.testing.assert_series_equal(expected[1], got[1].to_pandas())
