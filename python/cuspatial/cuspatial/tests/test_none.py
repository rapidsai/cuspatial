import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

import cuspatial


def test_empty_from_geopandas():
    gpdempty = gpd.GeoSeries([None])
    empty = cuspatial.from_geopandas(gpdempty)
    pd.testing.assert_series_equal(gpdempty, empty.to_geopandas())


def test_mix_from_geopandas():
    gpdempty = gpd.GeoSeries([None, Point(0, 1), None])
    empty = cuspatial.from_geopandas(gpdempty)
    pd.testing.assert_series_equal(gpdempty, empty.to_geopandas())


def test_first_from_geopandas():
    gpdempty = gpd.GeoSeries([Point(0, 1), None, None])
    empty = cuspatial.from_geopandas(gpdempty)
    pd.testing.assert_series_equal(gpdempty, empty.to_geopandas())


def test_last_from_geopandas():
    gpdempty = gpd.GeoSeries([None, None, Point(0, 1)])
    empty = cuspatial.from_geopandas(gpdempty)
    pd.testing.assert_series_equal(gpdempty, empty.to_geopandas())


def test_middle_from_geopandas():
    gpdempty = gpd.GeoSeries(
        [
            Polygon(((-8, -8), (-8, 8), (8, 8), (8, -8))),
            Polygon(((-2, -2), (-2, 2), (2, 2), (2, -2))),
            None,
            Polygon(((-10, -2), (-10, 2), (-6, 2), (-6, -2))),
            None,
            Polygon(((-2, 8), (-2, 12), (2, 12), (2, 8))),
            Polygon(((6, 0), (8, 2), (10, 0), (8, -2))),
            None,
            Polygon(((-2, -8), (-2, -4), (2, -4), (2, -8))),
        ]
    )
    empty = cuspatial.from_geopandas(gpdempty)
    pd.testing.assert_series_equal(gpdempty, empty.to_geopandas())


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
