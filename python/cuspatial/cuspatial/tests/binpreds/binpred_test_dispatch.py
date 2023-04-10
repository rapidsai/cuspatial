# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest
from shapely.geometry import LineString, MultiPoint, Point, Polygon

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
        (Point, MultiPoint),
        (Point, LineString),
        (Point, Polygon),
        (MultiPoint, Point),
        (MultiPoint, MultiPoint),
        (MultiPoint, LineString),
        (MultiPoint, Polygon),
        (LineString, Point),
        (LineString, MultiPoint),
        (LineString, LineString),
        (LineString, Polygon),
        (Polygon, Point),
        (Polygon, MultiPoint),
        (Polygon, LineString),
        (Polygon, Polygon),
    ]
)
def geotype_tuple(request):
    return request.param


"""The collection of test types. This section is dispatched based
on the feature type. Each feature pairing has a specific set of
comparisons that need to be performed to cover the entire test
space. This section will be replaced with specific feature
representations that cover all possible geometric combinations."""


@pytest.fixture(
    params=[
        "single_equal",
        "single_disjoint",
        "triple_center_equal",
        "triple_center_disjoint",
        "border_overlap",
        "interior_overlap",
    ]
)
def test_type(request):
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
    -|\--  # noqa: W605
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
    "polygon-in-point-point": (
        """
    x----
    |\  |  # noqa: W605
    | x |
    |/  |
    x----
    """,
        Polygon([(0.0, 0.0), (0.0, 1.0), (0.5, 0.5)]),
        point_polygon,
    ),
    "polygon-contained": (
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

"""Named features for each feature dispatch."""
points_dispatch = {
    "feature1": Point(0.0, 0.0),
    "feature2": Point(1.0, 1.0),
    "feature1_bo": Point(1.0, 1.0),
    "feature2_bo": Point(0.0, 0.0),
    "feature1_io": Point(1.0, 1.0),
    "feature2_io": Point(2.0, 2.0),
}

multipoints_dispatch = {
    "feature1": MultiPoint([(0.0, 0.0), (1.0, 1.0)]),
    "feature2": MultiPoint([(2.0, 2.0), (3.0, 3.0)]),
    "feature1_bo": MultiPoint([(0.0, 0.0), (1.0, 1.0)]),
    "feature2_bo": MultiPoint([(1.0, 1.0), (2.0, 2.0)]),
    "feature1_io": MultiPoint([(1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]),
    "feature2_io": MultiPoint([(3.0, 3.0), (1.0, 1.0), (4.0, 4.0)]),
}

linestrings_dispatch = {
    "feature1": LineString([(0.0, 0.0), (1.0, 1.0)]),
    "feature2": LineString([(2.0, 2.0), (3.0, 3.0)]),
    "feature1_bo": LineString([(0.0, 0.0), (1.0, 1.0)]),
    "feature2_bo": LineString([(1.1, 1.1), (2.0, 1.0)]),
    "feature1_io": LineString(
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    ),
    "feature2_io": LineString(
        [(4.0, 4.0), (1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]
    ),
}

polygons_dispatch = {
    "feature1": Polygon([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)]),
    "feature2": Polygon([(2.0, 2.0), (3.0, 3.0), (3.0, 2.0)]),
    "feature1_bo": Polygon([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)]),
    "feature2_bo": Polygon([(2.0, 2.0), (1.0, 1.0), (1.0, 0.0)]),
    "feature1_io": Polygon(
        [(4.0, 4.0), (4.0, -4.0), (-4.0, -4.0), (-4.0, 4.0)]
    ),
    "feature2_io": Polygon(
        [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)]
    ),
}

feature_dispatch = {
    Point: points_dispatch,
    MultiPoint: multipoints_dispatch,
    LineString: linestrings_dispatch,
    Polygon: polygons_dispatch,
}


"""Feature type dispatch function."""


def get_feature(feature_type, feature_name):
    return feature_dispatch[feature_type][feature_name]


"""Test type dispatch functions."""


def single_equal(type0, type1):
    return (
        cuspatial.GeoSeries([get_feature(type0, "feature1")]),
        cuspatial.GeoSeries([get_feature(type1, "feature2")]),
    )


def single_disjoint(type0, type1):
    return (
        cuspatial.GeoSeries([get_feature(type0, "feature1")]),
        cuspatial.GeoSeries([get_feature(type1, "feature2")]),
    )


def triple_center_equal(type0, type1):
    return (
        cuspatial.GeoSeries(
            [
                get_feature(type0, "feature1"),
                get_feature(type0, "feature2"),
                get_feature(type0, "feature1"),
            ]
        ),
        cuspatial.GeoSeries(
            [
                get_feature(type1, "feature2"),
                get_feature(type1, "feature2"),
                get_feature(type1, "feature2"),
            ]
        ),
    )


def triple_center_disjoint(type0, type1):
    return (
        cuspatial.GeoSeries(
            [
                get_feature(type0, "feature1"),
                get_feature(type0, "feature2"),
                get_feature(type0, "feature1"),
            ]
        ),
        cuspatial.GeoSeries(
            [
                get_feature(type1, "feature1"),
                get_feature(type1, "feature1"),
                get_feature(type1, "feature1"),
            ]
        ),
    )


def border_overlap(type0, type1):
    return (
        cuspatial.GeoSeries([get_feature(type0, "feature1_bo")]),
        cuspatial.GeoSeries([get_feature(type1, "feature2_bo")]),
    )


def interior_overlap(type0, type1):
    return (
        cuspatial.GeoSeries([get_feature(type0, "feature1_io")]),
        cuspatial.GeoSeries([get_feature(type1, "feature2_io")]),
    )


"""Dictionary for dispatching test types to functions that return
test data."""
predicate_dispatch = {
    "single_equal": single_equal,
    "single_disjoint": single_disjoint,
    "triple_center_equal": triple_center_equal,
    "triple_center_disjoint": triple_center_disjoint,
    "border_overlap": border_overlap,
    "interior_overlap": interior_overlap,
}


"""Dispatch function for test types."""


def feature_test_dispatch(type0, type1, test_type):
    dispatch_function = predicate_dispatch[test_type]
    return dispatch_function(type0, type1)
