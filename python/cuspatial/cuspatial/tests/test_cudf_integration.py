import numpy as np
import pandas as pd

import cuspatial


def test_sort_index_series(gs):
    gs.index = np.random.permutation(len(gs))
    cugs = cuspatial.from_geopandas(gs)
    pd.testing.assert_series_equal(
        gs.sort_index(), cugs.sort_index().to_pandas()
    )


def test_sort_index_dataframe(gpdf):
    gpdf.index = np.random.permutation(len(gpdf))
    cugpdf = cuspatial.from_geopandas(gpdf)
    pd.testing.assert_frame_equal(
        gpdf.sort_index(), cugpdf.sort_index().to_pandas()
    )


def test_sort_values(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    sort_gpdf = gpdf.sort_values("random")
    sort_cugpdf = cugpdf.sort_values("random").to_pandas()
    pd.testing.assert_frame_equal(sort_gpdf, sort_cugpdf)


def test_groupby(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    pd.testing.assert_frame_equal(
        gpdf.groupby("key")[["integer", "random"]].min().sort_index(),
        cugpdf.groupby("key")[["integer", "random"]]
        .min()
        .sort_index()
        .to_pandas(),
    )
