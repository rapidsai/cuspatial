# Copyright 2022 NVIDIA Corporation

import cupy as cp
import geopandas as gpd
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Polygon,
)

import cuspatial


@pytest.mark.parametrize(
    "range, expected",
    [[slice(0, 3), [0, 3, 4, 5]], [slice(3, 6), [0, 30, 40, 41]]],
)
def test_GeoColumnAccessor_polygon_offset(range, expected):
    gpdf = cuspatial.from_geopandas(
        gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    )
    shorter = gpdf[range]["geometry"]
    expected = cp.array(expected)
    got = shorter.polygons.geometry_offset
    assert cp.array_equal(got, expected)


@pytest.mark.parametrize(
    "range, expected",
    [
        [[0, 2], [0, 2, 5]],
        [[0, 1, 2], [0, 2, 4, 7]],
        [[2, 4], [0, 3, 6]],
        [[4, 5], [0, 3, 5]],
        [[4, 3, 2], [0, 3, 5, 8]],
    ],
)
def test_multipoint(range, expected):
    gs = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                MultiPoint([(0, 1), (0, 2)]),
                MultiPoint([(0, 3), (0, 4)]),
                MultiPoint([(0, 5), (0, 6), (0, 7)]),
                MultiPoint([(0, 8), (0, 9)]),
                MultiPoint([(0, 10), (0, 11), (0, 12)]),
                MultiPoint([(0, 13), (0, 14)]),
            ]
        )
    )
    t1 = gs[range]
    got = t1.multipoints.geometry_offset
    expected = cp.array(expected)
    assert cp.array_equal(got, expected)


@pytest.mark.parametrize(
    "range, expected",
    [
        [[0, 2], [0, 1, 3]],
        [[0, 1, 2], [0, 1, 2, 4]],
        [[2, 4], [0, 2, 4]],
        [[4, 5], [0, 2, 3]],
        [[4, 3, 2], [0, 2, 3, 5]],
    ],
)
def test_multilines_geometry_offset(range, expected):
    gs = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                LineString([(0, 1), (0, 2)]),
                LineString([(0, 3), (0, 4)]),
                MultiLineString(
                    [
                        LineString([(0, 5), (0, 6)]),
                        LineString([(0, 7), (0, 8)]),
                    ]
                ),
                LineString([(0, 9), (0, 10)]),
                MultiLineString(
                    [
                        LineString([(0, 11), (0, 12)]),
                        LineString([(0, 13), (0, 14)]),
                    ]
                ),
                LineString([(0, 15), (0, 16)]),
            ]
        )
    )
    t1 = gs[range]
    got = t1.lines.geometry_offset
    expected = cp.array(expected)
    assert cp.array_equal(got, expected)


@pytest.mark.parametrize(
    "range, expected",
    [
        [[0, 2], [0, 2, 4, 6]],
        [[0, 1, 2], [0, 2, 4, 6, 8]],
        [[2, 4], [0, 2, 4, 7, 9]],
        [[4, 5], [0, 3, 5, 7]],
        [[4, 3, 2], [0, 3, 5, 7, 9, 11]],
    ],
)
def test_multilines_part_offset(range, expected):
    gs = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                LineString([(0, 1), (0, 2)]),
                LineString([(0, 3), (0, 4)]),
                MultiLineString(
                    [
                        LineString([(0, 5), (0, 6)]),
                        LineString([(0, 7), (0, 8)]),
                    ]
                ),
                LineString([(0, 9), (0, 10)]),
                MultiLineString(
                    [
                        LineString([(0, 11), (0, 12), (0, 13)]),
                        LineString([(0, 14), (0, 15)]),
                    ]
                ),
                LineString([(0, 16), (0, 17)]),
            ]
        )
    )
    t1 = gs[range]
    got = t1.lines.part_offset
    expected = cp.array(expected)
    assert cp.array_equal(got, expected)


@pytest.mark.parametrize(
    "range, expected",
    [
        [[0, 2], [0, 1, 3]],
        [[0, 1, 2], [0, 1, 2, 4]],
        [[2, 4], [0, 2, 4]],
        [[4, 5], [0, 2, 3]],
        [[4, 3, 2], [0, 2, 3, 5]],
    ],
)
def test_multipolygons_geometry_offset(range, expected):
    gs = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                Polygon([(0, 1), (0, 2), (0, 18)]),
                Polygon([(0, 3), (0, 4), (0, 19)]),
                MultiPolygon(
                    [
                        Polygon([(0, 5), (0, 6), (0, 20)]),
                        Polygon([(0, 7), (0, 8), (0, 21)]),
                    ]
                ),
                Polygon([(0, 9), (0, 10), (0, 22)]),
                MultiPolygon(
                    [
                        Polygon([(0, 11), (0, 12), (0, 13), (0, 22)]),
                        Polygon([(0, 14), (0, 15), (0, 23)]),
                    ]
                ),
                Polygon([(0, 16), (0, 17), (0, 24)]),
            ]
        )
    )
    t1 = gs[range]
    got = t1.polygons.geometry_offset
    expected = cp.array(expected)
    assert cp.array_equal(got, expected)


@pytest.mark.parametrize(
    "range, expected",
    [
        [[0, 2], [0, 1, 2, 3]],
        [[0, 1, 2], [0, 1, 2, 3, 4]],
        [[2, 4], [0, 1, 2, 3, 4]],
        [[4, 5], [0, 1, 2, 3]],
        [[4, 3, 2], [0, 1, 2, 3, 4, 5]],
    ],
)
def test_multipolygons_part_offset(range, expected):
    gs = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                Polygon([(0, 1), (0, 2), (0, 18)]),
                Polygon([(0, 3), (0, 4), (0, 19)]),
                MultiPolygon(
                    [
                        Polygon([(0, 5), (0, 6), (0, 20)]),
                        Polygon([(0, 7), (0, 8), (0, 21)]),
                    ]
                ),
                Polygon([(0, 9), (0, 10), (0, 22)]),
                MultiPolygon(
                    [
                        Polygon([(0, 11), (0, 12), (0, 13), (0, 22)]),
                        Polygon([(0, 14), (0, 15), (0, 23)]),
                    ]
                ),
                Polygon([(0, 16), (0, 17), (0, 24)]),
            ]
        )
    )
    t1 = gs[range]
    got = t1.polygons.part_offset
    expected = cp.array(expected)
    assert cp.array_equal(got, expected)


@pytest.mark.parametrize(
    "range, expected",
    [
        [[0, 2], [0, 4, 8, 12]],
        [[0, 1, 2], [0, 4, 8, 12, 16]],
        [[2, 4], [0, 4, 8, 13, 17]],
        [[4, 5], [0, 5, 9, 13]],
        [[4, 3, 2], [0, 5, 9, 13, 17, 21]],
    ],
)
def test_multipolygons_ring_offset(range, expected):
    gs = cuspatial.from_geopandas(
        gpd.GeoSeries(
            [
                Polygon([(0, 1), (0, 2), (0, 18), (0, 1)]),
                Polygon([(0, 3), (0, 4), (0, 19), (0, 3)]),
                MultiPolygon(
                    [
                        Polygon([(0, 5), (0, 6), (0, 20), (0, 5)]),
                        Polygon([(0, 7), (0, 8), (0, 21), (0, 7)]),
                    ]
                ),
                Polygon([(0, 9), (0, 10), (0, 22), (0, 9)]),
                MultiPolygon(
                    [
                        Polygon([(0, 11), (0, 12), (0, 13), (0, 22), (0, 11)]),
                        Polygon([(0, 14), (0, 15), (0, 23), (0, 14)]),
                    ]
                ),
                Polygon([(0, 16), (0, 17), (0, 24), (0, 16)]),
            ]
        )
    )
    t1 = gs[range]
    got = t1.polygons.ring_offset
    expected = cp.array(expected)
    assert cp.array_equal(got, expected)
