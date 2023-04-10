# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest
from shapely.geometry import LineString, Point, Polygon

import cuspatial

"""Test Dispatch"""

"""Testing all combinations of possible geometry types and
binary predicates is a complex task. This file contains the
basic fixtures for all geometry types that are required to
cover different possible test outcomes. This file also contains
the dispatching system that uses each fixture to generate
a test for each possible combination of geometry types and
binary predicates.
"""

"""The following fixtures are used to generate tests for
each possible combination of geometry types and binary
predicates. The fixtures are combined in the test function
in `test_binpreds_test_dispatch.py` to make the following
tuple: (predicate, geotype, geotype, expected_result). The
geotype fixtures are used to generate the first two
elements of the tuple. The predicate fixture is used to
generate the third element of the tuple. The expected_result
fixture is used to generate the fourth element of the tuple.
"""
"""The collection of all possible binary predicates"""


@pytest.fixture(
    params=[
        "contains_properly",
        "geom_equals",
        "intersects",
        "covers",
        "crosses",
        "disjoint",
        "overlaps",
        "touches",
        "within",
    ]
)
def predicate(request):
    return request.param


"""The collection of all possible geometry types"""


@pytest.fixture(
    params=[
        (Point, Point),
        (Point, LineString),
        (Point, Polygon),
        (LineString, Point),
        (LineString, LineString),
        (LineString, Polygon),
        (Polygon, Point),
        (Polygon, LineString),
        (Polygon, Polygon),
    ]
)
def geotype_tuple(request):
    return request.param


"""The fundamental set of tests. This section is dispatched based
on the feature type. Each feature pairing has a specific set of
comparisons that need to be performed to cover the entire test
space. This section will be replaced with specific feature
representations that cover all possible geometric combinations."""


point_polygon = Polygon([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)])
features = {
    "point-point-disjoint": (
        """Two points apart.""",
        Point(0.0, 0.0),
        Point(1.0, 0.0),
    ),
    "point-point-equal": (
        """Two points together.""",
        Point(0.0, 0.0),
        Point(0.0, 0.0),
    ),
    "point-linestring-disjoint": (
        """Point and linestring are disjoint.""",
        Point(0.0, 0.0),
        LineString([(1.0, 0.0), (2.0, 0.0)]),
    ),
    "point-linestring-point": (
        """Point and linestring share a point.""",
        Point(0.0, 0.0),
        LineString([(0.0, 0.0), (2.0, 0.0)]),
    ),
    "point-linestring-edge": (
        """Point and linestring intersect.""",
        Point(0.5, 0.0),
        LineString([(0.0, 0.0), (1.0, 0.0)]),
    ),
    "point-polygon-disjoint": (
        """Point and polygon are disjoint.""",
        Point(-0.5, 0.5),
        point_polygon,
    ),
    "point-polygon-point": (
        """Point and polygon share a point.""",
        Point(0.0, 0.0),
        point_polygon,
    ),
    "point-polygon-edge": (
        """Point and polygon intersect.""",
        Point(0.5, 0.0),
        point_polygon,
    ),
    "point-polygon-in": (
        """Point is in polygon interior.""",
        Point(0.5, 0.5),
        point_polygon,
    ),
    "linestring-linestring-disjoint": (
        """
    x---x

    x---x
    """,
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.0, 1.0), (1.0, 1.0)]),
    ),
    "linestring-linestring-same": (
        """
    x---x
    """,
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.0, 0.0), (1.0, 0.0)]),
    ),
    "linestring-linestring-touches": (
        """
    x
    |
    |
    |
    x---x
    """,
        LineString([(0.0, 0.0), (0.0, 1.0)]),
        LineString([(0.0, 0.0), (1.0, 0.0)]),
    ),
    "linestring-linestring-crosses": (
        """
      x
      |
    x-|-x
      |
      x
    """,
        LineString([(0.5, 0.0), (0.5, 1.0)]),
        LineString([(0.0, 0.5), (1.0, 0.5)]),
    ),
    "linestring-polygon-disjoint": (
        """
    point_polygon above is drawn as
    -----
    |   |
    |   |
    |   |
    -----
    and the corresponding linestring is drawn as
    x---x
    or
    x
    |
    |
    |
    x
    """
        """
    x -----
    | |   |
    | |   |
    | |   |
    x -----
    """,
        LineString([(-0.5, 0.0), (-0.5, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-touch-point": (
        """
    x---x----
        |   |
        |   |
        |   |
        -----
    """,
        LineString([(-1.0, 0.0), (0.0, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-touch-edge": (
        """
        -----
        |   |
    x---x   |
        |   |
        -----
    """,
        LineString([(-1.0, 0.5), (0.0, 0.5)]),
        point_polygon,
    ),
    "linestring-polygon-overlap-edge": (
        """
    x----
    |   |
    |   |
    |   |
    x----
    """,
        LineString([(0.0, 0.0), (0.0, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-intersect-edge": (
        """
      -----
      |   |
      |   |
      |   |
    x---x--
    """,
        LineString([(-0.5, 0.0), (0.5, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-intersect-inner-edge": (
        """
    -----
    x   |
    |   |
    x   |
    -----

    The linestring in this case is shorter than the corners of the polygon.
    """,
        LineString([(0.25, 0.0), (0.75, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-point-interior": (
        """
    ----x
    |  /|
    | / |
    |/  |
    x----
    """,
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-edge-interior": (
        """
    --x--
    | | |
    | | |
    | | |
    --x--
    """,
        LineString([(0.5, 0.0), (0.5, 1.0)]),
        point_polygon,
    ),
    "linestring-polygon-in": (
        """
    -----
    | x |
    | | |
    | x |
    -----
    """,
        LineString([(0.5, 0.25), (0.5, 0.75)]),
        point_polygon,
    ),
    "linestring-polygon-in-out": (
        """
    -----
    |   |
    | x |
    | | |
    --|--
      |
      x
    """,
        LineString([(0.5, 0.5), (0.5, -0.5)]),
        point_polygon,
    ),
    "polygon-polygon-disjoint": (
        """
    Polygon polygon tests use a triangle for the lhs and a square for the rhs.
    The triangle is drawn as
    x---x
    |  /
    | /
    |/
    x

    The square is drawn as

    -----
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 2.0), (0.0, 3.0), (1.0, 3.0)]),
        point_polygon,
    ),
    "polygon-polygon-touch-point": (
        """
    x---x
    |  /
    | /
    |/
    x----
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-touch-edge": (
        """
     x---x
     |  /
     | /
     |/
    -x--x
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.25, 1.0), (0.25, 2.0), (1.25, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-overlap-edge": (
        """
    x
    |\
    | \
    |  \
    x---x
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-point-inside": (
        """
      x---x
      |  /
      | /
    --|/-
    | x |
    |   |
    |   |
    -----
    """,
        Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5)]),
        point_polygon,
    ),
    "polygon-polygon-point-outside": (
        """
     x
    -|\\-- # Double backslash due to issues with escape sequences
    |x-x|
    |   |
    |   |
    -----
    """,
        Polygon([(0.25, 0.75), (0.25, 1.25), (0.75, 0.75)]),
        point_polygon,
    ),
    "polygon-polygon-in-out-point": (
        """
      x
      |\
    --|-x
    | |/|
    | x |
    |   |
    x----
    """,
        Polygon([(0.5, 0.5), (0.5, 1.5), (1.0, 1.0)]),
        point_polygon,
    ),
    "polygon-polygon-in-point-point": (
        """
    x----
    |\\  | # Double backslash due to issues with escape sequences
    | x |
    |/  |
    x----
    """,
        Polygon([(0.0, 0.0), (0.0, 1.0), (0.5, 0.5)]),
        point_polygon,
    ),
    "polygon-polygon-contained": (
        """
    -----
    |  x|
    | /||
    |x-x|
    -----
    """,
        Polygon([(0.25, 0.25), (0.75, 0.75), (0.75, 0.25)]),
        point_polygon,
    ),
}

point_point_dispatch_list = [
    "point-point-disjoint",
    "point-point-equal",
]

point_linestring_dispatch_list = [
    "point-linestring-disjoint",
    "point-linestring-point",
    "point-linestring-edge",
]

point_polygon_dispatch_list = [
    "point-polygon-disjoint",
    "point-polygon-point",
    "point-polygon-edge",
    "point-polygon-in",
]

linestring_linestring_dispatch_list = [
    "linestring-linestring-disjoint",
    "linestring-linestring-same",
    "linestring-linestring-touches",
    "linestring-linestring-crosses",
]

linestring_polygon_dispatch_list = [
    "linestring-polygon-disjoint",
    "linestring-polygon-touch-point",
    "linestring-polygon-touch-edge",
    "linestring-polygon-overlap-edge",
    "linestring-polygon-intersect-edge",
    "linestring-polygon-intersect-inner-edge",
    "linestring-polygon-point-interior",
    "linestring-polygon-edge-interior",
    "linestring-polygon-in",
]

polygon_polygon_dispatch_list = [
    "polygon-polygon-disjoint",
    "polygon-polygon-touch-point",
    "polygon-polygon-touch-edge",
    "polygon-polygon-overlap-edge",
    "polygon-polygon-point-inside",
    "polygon-polygon-point-outside",
    "polygon-polygon-in-out-point",
    "polygon-polygon-in-point-point",
    "polygon-polygon-contained",
]


def object_dispatch(name_list):
    while True:
        # forward order
        for name in name_list:
            yield (name, features[name][1], features[name][2])
        # reversed order
        for name in name_list:
            yield (name, features[name][2], features[name][1])


type_dispatch = {
    (Point, Point): object_dispatch(point_point_dispatch_list),
    (Point, LineString): object_dispatch(point_linestring_dispatch_list),
    (LineString, Point): object_dispatch(point_linestring_dispatch_list),
    (Point, Polygon): object_dispatch(point_polygon_dispatch_list),
    (Polygon, Point): object_dispatch(point_polygon_dispatch_list),
    (LineString, LineString): object_dispatch(
        linestring_linestring_dispatch_list
    ),
    (LineString, Polygon): object_dispatch(linestring_polygon_dispatch_list),
    (Polygon, LineString): object_dispatch(linestring_polygon_dispatch_list),
    (Polygon, Polygon): object_dispatch(polygon_polygon_dispatch_list),
}


"""Feature type dispatch function."""


def feature_dispatch(types):
    generator = type_dispatch[types]
    yield next(generator)


"""Test type dispatch functions."""


def single_same(feature_0, feature_1):
    return (
        cuspatial.GeoSeries([feature_0]),
        cuspatial.GeoSeries([feature_0]),
    )


def single_different(feature_0, feature_1):
    return (
        cuspatial.GeoSeries([feature_0]),
        cuspatial.GeoSeries([feature_1]),
    )


def triple_center_same(feature_0, feature_1):
    return (
        cuspatial.GeoSeries(
            [
                feature_0,
                feature_1,
                feature_0,
            ]
        ),
        cuspatial.GeoSeries(
            [
                feature_1,
                feature_1,
                feature_1,
            ]
        ),
    )


def triple_center_different(feature_0, feature_1):
    return (
        cuspatial.GeoSeries(
            [
                feature_0,
                feature_1,
                feature_0,
            ]
        ),
        cuspatial.GeoSeries(
            [
                feature_0,
                feature_0,
                feature_0,
            ]
        ),
    )


"""Dispatch function for test types."""


def feature_test_dispatch(type_tuple):
    features = [*feature_dispatch(type_tuple)][0]
    result = single_different(features[1], features[2])
    return result
