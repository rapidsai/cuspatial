# Copyright (c) 2022 NVIDIA CORPORATION.
import geopandas as gpd
import numpy as np
import pandas as pd

import cuspatial


def test_sort_index_series(gs):
    gs.index = np.random.permutation(len(gs))
    cugs = cuspatial.from_geopandas(gs)
    expected = gs.sort_index()
    got = cugs.sort_index().to_pandas()
    gpd.testing.assert_geoseries_equal(got, expected)


def test_sort_index_dataframe(gpdf):
    gpdf.index = np.random.permutation(len(gpdf))
    cugpdf = cuspatial.from_geopandas(gpdf)
    expected = gpdf.sort_index()
    got = cugpdf.sort_index().to_pandas()
    gpd.testing.assert_geodataframe_equal(got, expected)


def test_sort_values(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    expected = gpdf.sort_values("random")
    got = cugpdf.sort_values("random").to_pandas()
    gpd.testing.assert_geodataframe_equal(got, expected)


def test_groupby(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    expected = gpdf.groupby("key")[["integer", "random"]].min().sort_index()
    got = (
        cugpdf.groupby("key")[["integer", "random"]]
        .min()
        .sort_index()
        .to_pandas()
    )
    pd.testing.assert_frame_equal(got, expected)
