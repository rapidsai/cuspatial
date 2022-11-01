# Copyright (c) 2022 NVIDIA CORPORATION.

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import cuspatial


def test_align():
    gpdpdf = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
        ]
    )
    gpdshort = gpdpdf.iloc[0:1]
    pdf = cuspatial.from_geopandas(gpdpdf)
    short = cuspatial.from_geopandas(gpdshort)
    gpdaligned = gpdshort.align(gpdpdf)
    shortaligned = short.align(pdf)
    pd.testing.assert_series_equal(gpdaligned[0], shortaligned[0].to_pandas())
    pd.testing.assert_series_equal(gpdaligned[1], shortaligned[1].to_pandas())


def test_align_reorder():
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
    gpdaligned = gpdalign.align(gpdreordered)
    aligned = align.align(reordered)
    pd.testing.assert_series_equal(gpdaligned[0], aligned[0].to_pandas())
    pd.testing.assert_series_equal(gpdaligned[1], aligned[1].to_pandas())
    gpdaligned = gpdreordered.align(gpdalign)
    aligned = reordered.align(align)
    pd.testing.assert_series_equal(gpdaligned[0], aligned[0].to_pandas())
    pd.testing.assert_series_equal(gpdaligned[1], aligned[1].to_pandas())
