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
    assert cugs.points.xy.sum().as_py() == 18
    assert cugs.lines.xy.sum().as_py() == 540
    assert cugs.multipoints.xy.sum().as_py() == 36
    assert cugs.polygons.xy.sum().as_py() == 7436
    assert cugs.polygons.polys.sum().as_py() == 38
    assert cugs.polygons.rings.sum().as_py() == 327


def test_from_geopandas_point():
    gs = gpd.GeoSeries(Point(1.0, 2.0))
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing._utils.assert_eq(
        cugs.points.xy, cudf.Series([1.0, 2.0], dtype="float64").to_arrow()
    )


def test_from_geopandas_multipoint():
    gs = gpd.GeoSeries(MultiPoint([(1.0, 2.0), (3.0, 4.0)]))
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing._utils.assert_eq(
        cugs.multipoints.xy, cudf.Series([1.0, 2.0, 3.0, 4.0], dtype="float64")
    )
    cudf.testing._utils.assert_eq(
        cugs.multipoints.offsets, cudf.Series([0, 2], dtype="int32")
    )


def test_from_geopandas_linestring():
    gs = gpd.GeoSeries(LineString(((4.0, 3.0), (2.0, 1.0))))
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing._utils.assert_eq(
        cugs.lines.xy, cudf.Series([4.0, 3.0, 2.0, 1.0], dtype="float64")
    )
    cudf.testing._utils.assert_eq(
        cugs.lines.offsets, cudf.Series([0, 2], dtype="int32")
    )


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
    cudf.testing._utils.assert_eq(
        cugs.lines.xy,
        cudf.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="float64"),
    )
    cudf.testing._utils.assert_eq(
        cugs.lines.offsets, cudf.Series([0, 2, 4], dtype="int32")
    )


def test_from_geopandas_polygon():
    gs = gpd.GeoSeries(
        Polygon(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing._utils.assert_eq(
        cugs.polygons.xy,
        cudf.Series([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype="float64"),
    )
    cudf.testing._utils.assert_eq(
        cugs.polygons.polys, cudf.Series([0, 1], dtype="int32").to_arrow()
    )
    cudf.testing._utils.assert_eq(
        cugs.polygons.rings, cudf.Series([0, 4], dtype="int32").to_arrow()
    )


def test_from_geopandas_polygon_hole():
    gs = gpd.GeoSeries(
        Polygon(
            ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
            [((1.0, 1.0), (1.0, 0.0), (0.0, 0.0))],
        )
    )
    cugs = cuspatial.from_geopandas(gs)
    cudf.testing._utils.assert_eq(
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
            ],
            dtype="float64",
        ),
    )
    cudf.testing._utils.assert_eq(
        cugs.polygons.polys, cudf.Series([0, 2], dtype="int32")
    )
    cudf.testing._utils.assert_eq(
        cugs.polygons.rings, cudf.Series([0, 4, 8], dtype="int32")
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
    cudf.testing._utils.assert_eq(
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
            ],
            dtype="float64",
        ),
    )
    cudf.testing._utils.assert_eq(
        cugs.polygons.polys, cudf.Series([0, 2], dtype="int32")
    )
    cudf.testing._utils.assert_eq(
        cugs.polygons.rings, cudf.Series([0, 4, 8], dtype="int32")
    )
