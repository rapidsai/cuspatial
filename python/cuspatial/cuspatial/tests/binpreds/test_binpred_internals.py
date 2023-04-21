# Copyright (c) 2020-2023, NVIDIA CORPORATION

import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point, Polygon

import cuspatial
from cuspatial.core.binpreds.binpred_dispatch import EQUALS_DISPATCH
from cuspatial.utils.binpred_utils import (
    _open_polygon_rings,
    _points_and_lines_to_multipoints,
)


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
    predicate = EQUALS_DISPATCH[
        (linestring1.column_type, linestring2.column_type)
    ]()
    got = predicate._reverse_linestrings(
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
    predicate = EQUALS_DISPATCH[
        (linestring1.column_type, linestring2.column_type)
    ]()
    got = predicate._reverse_linestrings(
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
    predicate = EQUALS_DISPATCH[
        (linestring1.column_type, linestring2.column_type)
    ]()
    got = predicate._reverse_linestrings(
        linestring1.lines.xy, linestring1.lines.part_offset
    ).to_pandas()
    expected = linestring2.lines.xy.to_pandas()
    pd.testing.assert_series_equal(got, expected)


def test_open_polygon_rings():
    polygon = cuspatial.GeoSeries(
        [
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    linestring = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, 0)]),
        ]
    )
    got = _open_polygon_rings(polygon)
    assert (got.lines.xy == linestring.lines.xy).all()


def test_open_polygon_rings_two():
    polygon = cuspatial.GeoSeries(
        [
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    linestring = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (1, 0)]),
            LineString([(0, 0), (1, 1), (1, 0)]),
        ]
    )
    got = _open_polygon_rings(polygon)
    assert (got.lines.xy == linestring.lines.xy).all()


def test_open_polygon_rings_three_varying_length():
    polygon = cuspatial.GeoSeries(
        [
            Polygon([(0, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
            Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
        ]
    )
    linestring = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (0, 1)]),
            LineString([(0, 0), (0, 1), (1, 1), (1, 0)]),
            LineString([(0, 0), (1, 1), (1, 0)]),
        ]
    )
    got = _open_polygon_rings(polygon)
    assert (got.lines.xy == linestring.lines.xy).all()


def test_points_and_lines_to_multipoints():
    mixed = cuspatial.GeoSeries(
        [
            Point(0, 0),
            LineString([(1, 1), (2, 2)]),
        ]
    )
    expected = cuspatial.GeoSeries(
        [
            MultiPoint([(0, 0)]),
            MultiPoint([(1, 1), (2, 2)]),
        ]
    )
    got = _points_and_lines_to_multipoints(mixed)
    assert (got.multipoints.xy == expected.multipoints.xy).all()


def test_points_and_lines_to_multipoints_reverse():
    mixed = cuspatial.GeoSeries(
        [
            LineString([(1, 1), (2, 2)]),
            Point(0, 0),
        ]
    )
    expected = cuspatial.GeoSeries(
        [
            MultiPoint([(1, 1), (2, 2)]),
            MultiPoint([(0, 0)]),
        ]
    )
    got = _points_and_lines_to_multipoints(mixed)
    assert (got.multipoints.xy == expected.multipoints.xy).all()


def test_points_and_lines_to_multipoints_two_points_one_linestring():
    mixed = cuspatial.GeoSeries(
        [
            Point(0, 0),
            LineString([(1, 1), (2, 2)]),
            Point(3, 3),
        ]
    )
    expected = cuspatial.GeoSeries(
        [
            MultiPoint([(0, 0)]),
            MultiPoint([(1, 1), (2, 2)]),
            MultiPoint([(3, 3)]),
        ]
    )
    got = _points_and_lines_to_multipoints(mixed)
    assert (got.multipoints.xy == expected.multipoints.xy).all()


def test_points_and_lines_to_multipoints_two_linestrings_one_point():
    mixed = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1)]),
            Point(2, 2),
            LineString([(3, 3), (4, 4)]),
        ]
    )
    expected = cuspatial.GeoSeries(
        [
            MultiPoint([(0, 0), (1, 1)]),
            MultiPoint([(2, 2)]),
            MultiPoint([(3, 3), (4, 4)]),
        ]
    )
    got = _points_and_lines_to_multipoints(mixed)
    assert (got.multipoints.xy == expected.multipoints.xy).all()


def test_points_and_lines_to_multipoints_complex():
    mixed = cuspatial.GeoSeries(
        [
            LineString([(0, 0), (1, 1), (2, 2), (3, 3)]),
            Point(4, 4),
            LineString([(5, 5), (6, 6)]),
            Point(7, 7),
            Point(8, 8),
            LineString([(9, 9), (10, 10), (11, 11)]),
            LineString([(12, 12), (13, 13)]),
            Point(14, 14),
        ]
    )
    expected = cuspatial.GeoSeries(
        [
            MultiPoint([(0, 0), (1, 1), (2, 2), (3, 3)]),
            MultiPoint([(4, 4)]),
            MultiPoint([(5, 5), (6, 6)]),
            MultiPoint([(7, 7)]),
            MultiPoint([(8, 8)]),
            MultiPoint([(9, 9), (10, 10), (11, 11)]),
            MultiPoint([(12, 12), (13, 13)]),
            MultiPoint([(14, 14)]),
        ]
    )
    got = _points_and_lines_to_multipoints(mixed)
    assert (got.multipoints.xy == expected.multipoints.xy).all()
