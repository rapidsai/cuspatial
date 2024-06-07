# Copyright (c) 2024, NVIDIA CORPORATION.

import geopandas as gpd
import pandas as pd
from geopandas.testing import assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import LineString, MultiLineString, Point

import cuspatial
from cuspatial.core.binops.intersection import pairwise_linestring_intersection


def run_test(s1, s2, expect_offset, expect_geom, expect_ids):
    offset, geoms, ids = pairwise_linestring_intersection(
        cuspatial.from_geopandas(s1), cuspatial.from_geopandas(s2)
    )

    assert_series_equal(
        expect_offset, pd.Series(offset.to_pandas()), check_dtype=False
    )
    assert_geoseries_equal(expect_geom, geoms.to_geopandas())
    assert_frame_equal(expect_ids, ids.to_pandas())


def test_empty():
    s1 = gpd.GeoSeries([])
    s2 = gpd.GeoSeries([])

    expect_offset = pd.Series([0], dtype="i4")
    expect_geom = gpd.GeoSeries([])
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [],
            "lhs_segment_id": [],
            "rhs_linestring_id": [],
            "rhs_segment_id": [],
        },
        dtype="i4",
    )

    run_test(s1, s2, expect_offset, expect_geom, expect_ids)


def test_one_pair():
    s1 = gpd.GeoSeries([LineString([(0, 0), (1, 1)])])
    s2 = gpd.GeoSeries([LineString([(0, 1), (1, 0)])])

    expect_offset = pd.Series([0, 1])
    expect_geom = s1.intersection(s2)
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0]],
            "lhs_segment_id": [[0]],
            "rhs_linestring_id": [[0]],
            "rhs_segment_id": [[0]],
        }
    )

    run_test(s1, s2, expect_offset, expect_geom, expect_ids)


def test_two_pairs():
    s1 = gpd.GeoSeries(
        [LineString([(0, 0), (1, 1)]), LineString([(0, 2), (2, 2), (2, 0)])]
    )
    s2 = gpd.GeoSeries(
        [
            LineString([(0, 1), (1, 0)]),
            LineString([(1, 1), (1, 3), (3, 3), (3, 1), (1.5, 1)]),
        ]
    )

    expect_offset = pd.Series([0, 1, 3])
    expect_geom = gpd.GeoSeries([Point(0.5, 0.5), Point(1, 2), Point(2, 1)])
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0], [0, 0]],
            "lhs_segment_id": [[0], [0, 1]],
            "rhs_linestring_id": [[0], [0, 0]],
            "rhs_segment_id": [[0], [0, 3]],
        }
    )

    run_test(s1, s2, expect_offset, expect_geom, expect_ids)


def test_one_pair_with_overlap():
    s1 = gpd.GeoSeries([LineString([(-1, 0), (0, 0), (0, 1), (-1, 1)])])
    s2 = gpd.GeoSeries([LineString([(1, 0), (0, 0), (0, 1), (1, 1)])])

    expect_offset = pd.Series([0, 1])
    expect_geom = s1.intersection(s2)
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0]],
            "lhs_segment_id": [[1]],
            "rhs_linestring_id": [[0]],
            "rhs_segment_id": [[1]],
        }
    )

    run_test(s1, s2, expect_offset, expect_geom, expect_ids)


def test_two_pairs_with_intersect_and_overlap():
    s1 = gpd.GeoSeries(
        [
            LineString([(-1, 0), (0, 0), (0, 1), (-1, 1)]),
            LineString([(-1, -1), (1, 1), (-1, 1)]),
        ]
    )
    s2 = gpd.GeoSeries(
        [
            LineString([(1, 0), (0, 0), (0, 1), (1, 1)]),
            LineString([(-1, 1), (1, -1), (-1, -1), (1, 1)]),
        ]
    )

    expect_offset = pd.Series([0, 1, 3])
    expect_geom = gpd.GeoSeries(
        [
            LineString([(0, 0), (0, 1)]),
            Point(-1, 1),
            LineString([(-1, -1), (1, 1)]),
        ]
    )
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0], [0, 0]],
            "lhs_segment_id": [[1], [1, 0]],
            "rhs_linestring_id": [[0], [0, 0]],
            "rhs_segment_id": [[1], [0, 2]],
        }
    )

    run_test(s1, s2, expect_offset, expect_geom, expect_ids)


def test_one_pair_multilinestring():
    s1 = gpd.GeoSeries(
        [MultiLineString([[(0, 0), (1, 1)], [(-1, 1), (0, 0)]])]
    )
    s2 = gpd.GeoSeries(
        [MultiLineString([[(0, 1), (1, 0)], [(-0.5, 0.5), (0, 0)]])]
    )

    expect_offset = pd.Series([0, 2])
    expect_geom = gpd.GeoSeries(
        [
            Point(0.5, 0.5),
            LineString([(-0.5, 0.5), (0, 0)]),
        ]
    )
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0, 1]],
            "lhs_segment_id": [[0, 0]],
            "rhs_linestring_id": [[0, 1]],
            "rhs_segment_id": [[0, 0]],
        }
    )

    run_test(s1, s2, expect_offset, expect_geom, expect_ids)


def test_three_pairs_identical_has_ring():
    lhs = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (1, 1)]),
        ]
    )
    rhs = gpd.GeoSeries(
        [
            LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
        ]
    )

    expect_offset = pd.Series([0, 2, 4, 6])
    expect_geom = gpd.GeoSeries(
        [
            Point(0, 0),
            Point(1, 1),
            Point(0, 0),
            Point(1, 1),
            Point(0, 0),
            Point(1, 1),
        ]
    )
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0, 0], [0, 0], [0, 0]],
            "lhs_segment_id": [[0, 0], [0, 0], [0, 0]],
            "rhs_linestring_id": [[0, 0], [0, 0], [0, 0]],
            "rhs_segment_id": [[0, 1], [0, 1], [0, 1]],
        }
    )

    run_test(lhs, rhs, expect_offset, expect_geom, expect_ids)


def test_three_pairs_identical_no_ring():
    lhs = gpd.GeoSeries(
        [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (1, 1)]),
        ]
    )
    rhs = gpd.GeoSeries(
        [
            LineString([(0, 0), (0, 1), (1, 1), (1, 0)]),
            LineString([(0, 0), (0, 1), (1, 1), (1, 0)]),
            LineString([(0, 0), (0, 1), (1, 1), (1, 0)]),
        ]
    )

    expect_offset = pd.Series([0, 2, 4, 6])
    expect_geom = gpd.GeoSeries(
        [
            Point(0, 0),
            Point(1, 1),
            Point(0, 0),
            Point(1, 1),
            Point(0, 0),
            Point(1, 1),
        ]
    )
    expect_ids = pd.DataFrame(
        {
            "lhs_linestring_id": [[0, 0], [0, 0], [0, 0]],
            "lhs_segment_id": [[0, 0], [0, 0], [0, 0]],
            "rhs_linestring_id": [[0, 0], [0, 0], [0, 0]],
            "rhs_segment_id": [[0, 1], [0, 1], [0, 1]],
        }
    )

    run_test(lhs, rhs, expect_offset, expect_geom, expect_ids)
