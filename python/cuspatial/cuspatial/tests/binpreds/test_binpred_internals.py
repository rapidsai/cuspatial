# Copyright (c) 2020-2023, NVIDIA CORPORATION

import pandas as pd
from shapely.geometry import LineString

import cuspatial


def test_internal_reversed_linestrings():
    linestring1 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    linestring2 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 0), (1, 1), (0, 0)]),
        ]
    )
    from cuspatial.core.binpreds.binpreds import EqualsBinpred

    bp = EqualsBinpred(linestring1, linestring2)
    got = bp._reverse_linestrings(
        linestring1.lines.xy, linestring1.lines.part_offset
    ).to_pandas()
    expected = linestring2.lines.xy.to_pandas()
    pd.testing.assert_series_equal(got, expected)


def test_internal_reversed_linestrings_pair():
    linestring1 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, 0), (0, 0)]),
            LineString([(0, 0), (1, 1), (1, 0)]),
        ]
    )
    linestring2 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 0), (1, 1), (0, 0)]),
            LineString([(1, 0), (1, 1), (0, 0)]),
        ]
    )
    from cuspatial.core.binpreds.binpreds import EqualsBinpred

    bp = EqualsBinpred(linestring1, linestring2)
    got = bp._reverse_linestrings(
        linestring1.lines.xy, linestring1.lines.part_offset
    ).to_pandas()
    expected = linestring2.lines.xy.to_pandas()
    pd.testing.assert_series_equal(got, expected)


def test_internal_reversed_linestrings_triple():
    linestring1 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, 0), (0, 0)]),
            LineString([(0, 0), (1, 1), (1, 0)]),
            LineString([(0, 0), (1, 1), (1, 0), (0, 0), (1, 1)]),
        ]
    )
    linestring2 = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 0), (1, 1), (0, 0)]),
            LineString([(1, 0), (1, 1), (0, 0)]),
            LineString([(1, 1), (0, 0), (1, 0), (1, 1), (0, 0)]),
        ]
    )
    from cuspatial.core.binpreds.binpreds import EqualsBinpred

    bp = EqualsBinpred(linestring1, linestring2)
    got = bp._reverse_linestrings(
        linestring1.lines.xy, linestring1.lines.part_offset
    ).to_pandas()
    expected = linestring2.lines.xy.to_pandas()
    pd.testing.assert_series_equal(got, expected)
