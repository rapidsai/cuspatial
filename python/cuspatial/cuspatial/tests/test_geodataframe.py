# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.affinity import rotate
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

np.random.seed(0)


def random_polygon(distance_from_origin):
    outer = Point(distance_from_origin * 2, 0).buffer(1)
    inners = []
    for i in range(np.random.randint(1, 4)):
        inner = Point(distance_from_origin + i * 0.1, 0).buffer(0.01)
        inners.append(inner)
    together = Polygon(outer, inners)
    result = rotate(together, np.random.random() * 2 * np.pi, use_radians=True)
    return result


def random_multipolygon(size):
    polygons = []
    for i in range(size):
        polygons.append(random_polygon(i))
    result = MultiPolygon(polygons)
    return result


def generator(size, has_z=False):
    obj_type = np.random.randint(1, 7)
    if obj_type == 1:
        return Point(np.random.random(2))
    else:
        if obj_type == 2:
            points = np.random.random(size * 2).reshape(size, 2)
            return MultiPoint(points)
        elif obj_type == 3:
            points = np.random.random(size * 2).reshape(size, 2)
            return LineString(points)
        elif obj_type == 4:
            num_lines = np.random.randint(3, np.ceil(np.sqrt(size)) + 3)
            points = np.random.random(num_lines * size * 2).reshape(
                num_lines, size, 2
            )
            return MultiLineString(tuple(points))
        elif obj_type == 5:
            return random_polygon(size)
        elif obj_type == 6:
            return random_multipolygon(size)


def assert_eq_point(p1, p2):
    assert type(p1) == type(p2)
    assert p1.x == p2.x
    assert p1.y == p2.y
    assert p1.has_z == p2.has_z
    if p1.has_z:
        assert p1.z == p2.z
    assert True


def assert_eq_multipoint(p1, p2):
    assert type(p1) == type(p2)
    assert len(p1) == len(p2)
    for i in range(len(p1)):
        assert_eq_point(p1[i], p2[i])


def assert_eq_polygon(p1, p2):
    if not p1.equals(p2):
        raise ValueError


def assert_eq_multipolygon(p1, p2):
    if not p1.equals(p2):
        raise ValueError


def assert_eq_geo_df(geo1, geo2):
    if type(geo1) != type(geo2):
        assert TypeError
    assert geo1.columns.equals(geo2.columns)
    for col in geo1.columns:
        if geo1[col].dtype == "geometry":
            assert geo1[col].equals(geo2[col])
        else:
            pd.testing.assert_series_equal(geo1[col], geo2[col])


def test_select_multiple_columns(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    pd.testing.assert_frame_equal(
        cugpdf[["geometry", "key"]].to_pandas(), gpdf[["geometry", "key"]]
    )


def test_type_persistence(gpdf):
    cugpdf = cuspatial.from_geopandas(gpdf)
    assert type(cugpdf["geometry"]) == cuspatial.GeoSeries


def test_interleaved_point(gpdf, polys):
    cugpdf = cuspatial.from_geopandas(gpdf)
    cugs = cugpdf["geometry"]
    gs = gpdf["geometry"]
    pd.testing.assert_series_equal(
        cugs.points.x.to_pandas().reset_index(drop=True),
        gs[gs.type == "Point"].x.reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        cugs.points.y.to_pandas().reset_index(drop=True),
        gs[gs.type == "Point"].y.reset_index(drop=True),
    )


def test_interleaved_multipoint(gpdf, polys):
    cugpdf = cuspatial.from_geopandas(gpdf)
    cugs = cugpdf["geometry"]
    gs = gpdf["geometry"]
    cudf.testing.assert_series_equal(
        cudf.Series.from_arrow(cugs.multipoints.x.to_arrow()),
        cudf.Series(
            np.array(
                [
                    np.array(p.__geo_interface__["coordinates"])[:, 0]
                    for p in gs[gs.type == "MultiPoint"]
                ]
            ).flatten()
        ),
    )
    cudf.testing.assert_series_equal(
        cudf.Series.from_arrow(cugs.multipoints.y.to_arrow()),
        cudf.Series(
            np.array(
                [
                    np.array(p.__geo_interface__["coordinates"])[:, 1]
                    for p in gs[gs.type == "MultiPoint"]
                ]
            ).flatten()
        ),
    )


def test_interleaved_lines(gpdf, polys):
    cugpdf = cuspatial.from_geopandas(gpdf)
    cugs = cugpdf["geometry"]
    cudf.testing.assert_series_equal(
        cudf.Series.from_arrow(cugs.lines.x.to_arrow()),
        cudf.Series(
            np.array([range(11, 34, 2)]).flatten(),
            dtype="float64",
        ),
    )
    cudf.testing.assert_series_equal(
        cudf.Series.from_arrow(cugs.lines.y.to_arrow()),
        cudf.Series(
            np.array([range(12, 35, 2)]).flatten(),
            dtype="float64",
        ),
    )


def test_interleaved_polygons(gpdf, polys):
    cugpdf = cuspatial.from_geopandas(gpdf)
    cugs = cugpdf["geometry"]
    cudf.testing.assert_series_equal(
        cudf.Series.from_arrow(cugs.polygons.x.to_arrow()),
        cudf.Series(polys[:, 0], dtype="float64"),
    )
    cudf.testing.assert_series_equal(
        cudf.Series.from_arrow(cugs.polygons.y.to_arrow()),
        cudf.Series(polys[:, 1], dtype="float64"),
    )


def test_to_geopandas_with_geopandas_dataset():
    df = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    gdf = cuspatial.from_geopandas(df)
    assert_eq_geo_df(df, gdf.to_geopandas())


def test_to_shapely_random():
    geos_list = []
    for i in range(250):
        geo = generator(3)
        geos_list.append(geo)
    gi = gpd.GeoDataFrame(
        {"geometry": geos_list, "integer": range(len(geos_list))}
    )
    cugpdf = cuspatial.from_geopandas(gi)
    cugpdf_back = cugpdf.to_geopandas()
    assert_eq_geo_df(gi, cugpdf_back)


@pytest.mark.parametrize(
    "pre_slice",
    [
        (slice(0, 12)),
        (slice(0, 10, 1)),
        (slice(0, 3, 1)),
        (slice(3, 6, 1)),
        (slice(6, 9, 1)),
    ],
)
def test_pre_slice(gpdf, pre_slice):
    geometries = gpdf.iloc[pre_slice, :]
    gi = gpd.GeoDataFrame(geometries)
    cugpdf = cuspatial.from_geopandas(gi)
    cugpdf_back = cugpdf.to_geopandas()
    assert_eq_geo_df(gi, cugpdf_back)


@pytest.mark.parametrize(
    "post_slice",
    [
        (slice(0, 12)),
        (slice(0, 10, 1)),
        (slice(0, 3, 1)),
        (slice(3, 6, 1)),
        (slice(6, 9, 1)),
    ],
)
def test_post_slice(gpdf, post_slice):
    geometries = gpdf
    gi = gpd.GeoDataFrame(geometries)
    cugpdf = cuspatial.from_geopandas(gi)
    cugpdf_back = cugpdf.to_geopandas()
    assert_eq_geo_df(gi[post_slice], cugpdf_back[post_slice])


@pytest.mark.parametrize(
    "inline_slice",
    [
        (slice(0, 12)),
        (slice(0, 10, 1)),
        (slice(0, 3, 1)),
        (slice(3, 6, 1)),
        (slice(6, 9, 1)),
    ],
)
def test_inline_slice(gpdf, inline_slice):
    gi = gpd.GeoDataFrame(gpdf)
    cugpdf = cuspatial.from_geopandas(gi)
    assert_eq_geo_df(gi[inline_slice], cugpdf[inline_slice].to_pandas())


def test_slice_column_order(gpdf):
    gi = gpd.GeoDataFrame(gpdf)
    cugpdf = cuspatial.from_geopandas(gi)

    slice_df = cuspatial.core.geodataframe.GeoDataFrame(
        {
            "geo1": cugpdf["geometry"],
            "data1": np.arange(len(cugpdf)),
            "geo2": cugpdf["geometry"],
            "data2": np.arange(len(cugpdf)),
        }
    )
    slice_gi = slice_df.to_pandas()
    assert_eq_geo_df(slice_gi[0:5], slice_df[0:5].to_pandas())

    slice_df = cuspatial.core.geodataframe.GeoDataFrame(
        {
            "data1": np.arange(len(cugpdf)),
            "geo1": cugpdf["geometry"],
            "geo2": cugpdf["geometry"],
            "data2": np.arange(len(cugpdf)),
        }
    )
    slice_gi = slice_df.to_pandas()
    assert_eq_geo_df(slice_gi[5:], slice_df[5:].to_pandas())

    slice_df = cuspatial.core.geodataframe.GeoDataFrame(
        {
            "data1": np.arange(len(cugpdf)),
            "geo4": cugpdf["geometry"],
            "data2": np.arange(len(cugpdf)),
            "geo3": cugpdf["geometry"],
            "data3": np.arange(len(cugpdf)),
            "geo2": cugpdf["geometry"],
            "geo1": cugpdf["geometry"],
            "data4": np.arange(len(cugpdf)),
            "data5": np.arange(len(cugpdf)),
            "data6": np.arange(len(cugpdf)),
        }
    )
    slice_gi = slice_df.to_pandas()
    assert_eq_geo_df(slice_gi[5:], slice_df[5:].to_pandas())


@pytest.mark.parametrize(
    "df_boolmask",
    [
        np.repeat(True, 12),
        np.repeat((np.repeat(True, 3), np.repeat(False, 3)), 2).flatten(),
        np.repeat(False, 12),
        np.repeat((np.repeat(False, 3), np.repeat(True, 3)), 2).flatten(),
        np.repeat([True, False], 6).flatten(),
    ],
)
def test_boolmask(gpdf, df_boolmask):
    geometries = gpdf
    gi = gpd.GeoDataFrame(geometries)
    cugpdf = cuspatial.from_geopandas(gi)
    cugpdf_back = cugpdf.to_geopandas()
    assert_eq_geo_df(gi[df_boolmask], cugpdf_back[df_boolmask])


def test_memory_usage(gs):
    assert gs.memory_usage() == 224
    host_dataframe = gpd.read_file(
        gpd.datasets.get_path("naturalearth_lowres")
    )
    gpu_dataframe = cuspatial.from_geopandas(host_dataframe)
    # The df size is 8kb of cudf rows and 217kb of the geometry column
    assert gpu_dataframe.memory_usage().sum() == 224945


def test_from_dict():
    p1 = Point([0, 1])
    p2 = Point([2, 3])
    p3 = Point([4, 5])
    p4 = MultiPoint([[6, 7], [8, 9]])
    gi = gpd.GeoDataFrame({"geometry": [p1, p2, p3, p4]})
    cu = cuspatial.GeoDataFrame({"geometry": [p1, p2, p3, p4]})
    assert_eq_geo_df(gi, cu.to_geopandas())


def test_from_dict2():
    points = {
        "a": [Point(0, 1), Point(2, 3)],
        "b": [MultiPoint([(4, 5), (6, 7)]), Point(8, 9)],
    }
    gpu_points_df = cuspatial.GeoDataFrame(points)
    assert (gpu_points_df["a"].points.xy == cudf.Series([0, 1, 2, 3])).all()
    assert (gpu_points_df["b"].points.xy == cudf.Series([8, 9])).all()
    assert (
        gpu_points_df["b"].multipoints.xy == cudf.Series([4, 5, 6, 7])
    ).all()


def test_from_gp_geoseries_dict():
    gp_geo_series = {
        "gpa": gpd.GeoSeries([Point(0, 1)]),
        "gpb": gpd.GeoSeries([MultiPoint([(2, 3), (4, 5)])]),
    }
    gp_df = gpd.GeoDataFrame(gp_geo_series)
    gpu_gp_df = cuspatial.GeoDataFrame(gp_geo_series)
    assert_eq_geo_df(gp_df, gpu_gp_df.to_geopandas())
    gp_df2 = gpd.GeoDataFrame({"gpdfa": gp_df["gpb"], "gpdfb": gp_df["gpa"]})
    gpdf = cuspatial.GeoDataFrame(
        {"gpdfa": gpu_gp_df["gpb"], "gpdfb": gpu_gp_df["gpa"]}
    )
    assert_eq_geo_df(gp_df2, gpdf.to_geopandas())


# Randomly collects 5 of 6 gpdf columns, slices them, and tries
# to create a new DataFrame from a dict based on those columns.
@pytest.mark.parametrize(
    "dict_slice",
    [
        (slice(0, 12)),
        (slice(0, 10, 1)),
        (slice(0, 3, 1)),
        (slice(3, 6, 1)),
        (slice(6, 9, 1)),
    ],
)
def test_from_dict_slices(gpdf, dict_slice):
    sliced = gpdf[dict_slice]
    sliced_dict = {
        char: sliced[col]
        for char, col in zip(
            np.array([*"abcdef"])[np.random.randint(0, 5, 5)], sliced.columns
        )
    }
    gpdf = gpd.GeoDataFrame(sliced_dict)
    cugpdf = cuspatial.GeoDataFrame(sliced_dict)
    assert_eq_geo_df(gpdf, cugpdf.to_geopandas())


def test_from_dict_with_list():
    dict_with_lists = {
        "a": [1, 2, 3, 4],
        "geometry": [
            Point(0, 1),
            Point(2, 3),
            MultiPoint([(4, 5), (6, 7)]),
            Point(8, 9),
        ],
        "c": [*"abcd"],
    }
    assert_eq_geo_df(
        gpd.GeoDataFrame(dict_with_lists),
        cuspatial.GeoDataFrame(dict_with_lists).to_geopandas(),
    )


@pytest.mark.parametrize("level", [None, 0, 1])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index(level, drop, inplace, col_level, col_fill):
    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    gpdf = gpd.GeoDataFrame(
        {
            "geometry": [
                Point(0, 1),
                Point(2, 3),
                MultiPoint([(4, 5), (6, 7)]),
                Point(8, 9),
            ],
            "a": [*"abcd"],
        },
        index=midx,
    )
    gdf = cuspatial.from_geopandas(gpdf)
    expected = gpdf.reset_index(level, drop, inplace, col_level, col_fill)
    got = gdf.reset_index(level, drop, inplace, col_level, col_fill)
    if inplace:
        expected = gpdf
        got = gdf
    pd.testing.assert_frame_equal(expected, got.to_pandas())
