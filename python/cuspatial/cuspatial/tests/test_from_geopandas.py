# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf

import cuspatial


def test_geobuffer_len(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert len(cugs._column._geo) == 12


def test_mixed_dataframe(gs):
    gpdf = gpd.GeoDataFrame({"a": list(range(100, 100 + len(gs))), "b": gs})
    cgdf = cuspatial.from_geopandas(gpdf)
    pd.testing.assert_series_equal(gpdf["a"], cgdf["a"].to_pandas())
    assert gpdf["b"].equals(cgdf["b"].to_pandas())
    pd.testing.assert_frame_equal(gpdf, cgdf.to_pandas())


def test_dataframe_column_access(gs):
    gpdf = gpd.GeoDataFrame({"a": list(range(0, len(gs))), "b": gs})
    cgdf = cuspatial.from_geopandas(gpdf)
    assert gpdf["b"].equals(cgdf["b"].to_pandas())


def test_from_geoseries_complex(gs):
    cugs = cuspatial.from_geopandas(gs)
    assert cugs.points.xy.sum() == 18
    assert cugs.lines.xy.sum() == 540
    assert cugs.multipoints.xy.sum() == 36
    assert cugs.polygons.xy.sum() == 7436
    assert cugs.polygons.polys.sum() == 38
    assert cugs.polygons.rings.sum() == 654


def test_from_geopandas_point():
    gs = gpd.GeoSeries(Point(1.0, 2.0))
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(cugs.points.xy, cudf.Series([1.0, 2.0]))


def test_from_geopandas_multipoint():
    gs = gpd.GeoSeries(MultiPoint([(1.0, 2.0), (3.0, 4.0)]))
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(
        cugs.multipoints.xy, cudf.Series([1.0, 2.0, 3.0, 4.0])
    )
    cudf.testing.assert_series_equal(
        cugs.multipoints.offsets, cudf.Series([0, 4])
    )


def test_from_geopandas_linestring():
    gs = gpd.GeoSeries(LineString(((4.0, 3.0), (2.0, 1.0))))
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(
        cugs.lines.xy, cudf.Series([4.0, 3.0, 2.0, 1.0])
    )
    cudf.testing.assert_series_equal(cugs.lines.offsets, cudf.Series([0, 4]))


def test_from_geopandas_multilinestring():
    gs = gpd.GeoSeries(
        MultiLineString(
            (
                ((1.0, 2.0), (3.0, 4.0)),
                ((5.0, 6.0), (7.0, 8.0)),
            )
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(
        cugs.lines.xy,
        cudf.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    cudf.testing.assert_series_equal(
        cugs.lines.offsets, cudf.Series([0, 4, 8])
    )


def test_from_geopandas_polygon():
    gs = gpd.GeoSeries(
        Polygon(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(
        cugs.polygons.xy,
        cudf.Series([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    cudf.testing.assert_series_equal(cugs.polygons.polys, cudf.Series([0, 1]))
    cudf.testing.assert_series_equal(cugs.polygons.rings, cudf.Series([0, 8]))


def test_from_geopandas_polygon_hole():
    gs = gpd.GeoSeries(
        Polygon(
            ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
            [((1.0, 1.0), (1.0, 0.0), (0.0, 0.0))],
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(
        cugs.polygons.xy,
        cudf.Series(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ]
        ),
    )
    cudf.testing.assert_series_equal(cugs.polygons.polys, cudf.Series([0, 2]))
    cudf.testing.assert_series_equal(
        cugs.polygons.rings, cudf.Series([0, 8, 16])
    )


def test_from_geopandas_multipolygon():
    gs = gpd.GeoSeries(
        MultiPolygon(
            [
                (
                    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
                    [((1.0, 1.0), (1.0, 0.0), (0.0, 0.0))],
                )
            ]
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing.assert_series_equal(
        cugs.polygons.xy,
        cudf.Series(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ]
        ),
    )
    cudf.testing.assert_series_equal(cugs.polygons.polys, cudf.Series([0, 2]))
    cudf.testing.assert_series_equal(
        cugs.polygons.rings, cudf.Series([0, 8, 16])
    )
