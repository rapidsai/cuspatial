# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest
from shapely.geometry import LineString, Point, Polygon

import cuspatial

"""Test Dispatch"""

"""This file is used to generate tests for all possible combinations
of geometry types and binary predicates. The tests are generated
using the fixtures defined in this file. The fixtures are combined
in the test function in `test_binpreds_test_dispatch.py` to make
a Tuple: (feature-name, feature-description, feature-lhs,
feature-rhs). The feature-name and feature-descriptions are not used
in the test but are used for development and debugging.
"""


@pytest.fixture(
    params=[
        "contains",
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
    """The collection of all supported binary predicates"""
    return request.param


"""The fundamental set of tests. This section is dispatched based
on the feature type. Each feature pairing has a specific set of
comparisons that need to be performed to cover the entire test
space. This section contains specific feature representations
that cover all possible geometric combinations."""


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
    "linestring-linestring-covers": (
        """
        x
       x
      /
     x
    x
    """,
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        LineString([(0.25, 0.25), (0.5, 0.5)]),
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
    "linestring-linestring-touch-interior": (
        """
    x   x
    |  /
    | /
    |/
    x---x
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.0, 0.0), (1.0, 1.0)]),
    ),
    "linestring-linestring-touch-edge": (
        """
      x
      |
      |
      |
    x-x-x
    """,
        LineString([(0.0, 0.0), (1.0, 0.0)]),
        LineString([(0.5, 0.0), (0.5, 1.0)]),
    ),
    "linestring-linestring-touch-edge-twice": (
        """
        x
       x
      / \\
     x---x
    x
    """,
        LineString([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]),
        LineString([(0.25, 0.25), (1.0, 0.0), (0.5, 0.5)]),
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
    "linestring-linestring-touch-and-cross": (
        """
        x
        |
        x
        |\\
      x---x
        x
    """,
        LineString([(0.0, 0.0), (1.0, 1.0)]),
        LineString([(0.5, 0.5), (1.0, 0.1), (-1.0, 0.1)]),
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
    "linestring-polygon-crosses": (
        """
      x
    --|--
    | | |
    | | |
    | | |
    --|--
      x
    """,
        LineString([(0.5, 1.25), (0.5, -0.25)]),
        point_polygon,
    ),
    "linestring-polygon-cross-concave-edge": (
        """
    x  x  x
    |\\ | /|
    | xx- |
    |  |  |
    ---x---
    """,
        LineString([(0.5, 0.0), (0.5, 1.0)]),
        Polygon([(0, 0), (0, 1), (0.3, 0.4), (1, 1), (1, 0)]),
    ),
    "linestring-polygon-half-in": (
        """
    -----
    |   |
    | x |
    |/ \\|
    xx-xx
    """,
        LineString(
            [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 0.0), (1.0, 0.0)]
        ),
        point_polygon,
    ),
    "linestring-polygon-half-out": (
        """
    -----
    |   |
    |   |
    |   |
    xx-xx
     \\/
      x
    """,
        LineString(
            [(0.0, 0.0), (0.25, 0.0), (0.5, -0.5), (0.75, 0.0), (1.0, 0.0)]
        ),
        point_polygon,
    ),
    "linestring-polygon-two-edges": (
        """
    x----
    |   |
    |   |
    |   |
    x---x
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.0, 0.0)]),
        point_polygon,
    ),
    "linestring-polygon-edge-to-interior": (
        """
    x----
    |   |
    |  -x
    |-/ |
    x----
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.0, 0.5)]),
        point_polygon,
    ),
    "linestring-polygon-edge-cross-to-exterior": (
        """
    x------
    |     |
    |    ---x
    | --- |
    x------
    """,
        LineString([(0.0, 1.0), (0.0, 0.0), (1.5, 0.5)]),
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
    |\\
    | \\
    |  \\
    x---x
    |   |
    |   |
    |   |
    -----
    """,
        Polygon([(0.0, 1.0), (0.0, 2.0), (1.0, 2.0)]),
        point_polygon,
    ),
    "polygon-polygon-overlap-inside-edge": (
        """
          x
         /|
    x---x |
    \\ /  |
      x   |
     /    |
    x-----x
    """,
        Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
        Polygon([(0.25, 0.25), (0.5, 0.5), (0, 0.5), (0.25, 0.25)]),
    ),
    "polygon-polygon-point-inside": (
        """
      x---x
      |  /
    --|-/
    | |/|
    | x |
    |   |
    -----
    """,
        Polygon([(0.5, 0.5), (0.5, 1.5), (1.5, 1.5)]),
        point_polygon,
    ),
    "polygon-polygon-point-outside": (
        """
     x
    -|\\--
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
      |\\
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
    |\\  |
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
    "polygon-polygon-same": (
        """
    x---x
    |   |
    |   |
    |   |
    x---x
    """,
        point_polygon,
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
    "linestring-linestring-covers",
    "linestring-linestring-touches",
    "linestring-linestring-touch-interior",
    "linestring-linestring-touch-edge",
    "linestring-linestring-touch-edge-twice",
    "linestring-linestring-crosses",
    "linestring-linestring-touch-and-cross",
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
    "linestring-polygon-crosses",
    "linestring-polygon-cross-concave-edge",
    "linestring-polygon-half-in",
    "linestring-polygon-half-out",
    "linestring-polygon-two-edges",
    "linestring-polygon-edge-to-interior",
    "linestring-polygon-edge-cross-to-exterior",
]

polygon_polygon_dispatch_list = [
    "polygon-polygon-disjoint",
    "polygon-polygon-touch-point",
    "polygon-polygon-touch-edge",
    "polygon-polygon-overlap-edge",
    "polygon-polygon-overlap-inside-edge",
    "polygon-polygon-point-inside",
    "polygon-polygon-point-outside",
    "polygon-polygon-in-out-point",
    "polygon-polygon-in-point-point",
    "polygon-polygon-contained",
    "polygon-polygon-same",
]


def object_dispatch(name_list):
    """Generate a list of test cases for a given set of test names."""
    for name in name_list:
        yield (name, features[name][0], features[name][1], features[name][2])


type_dispatch = {
    # A dictionary of test cases for each geometry type combination.
    # Still needs MultiPoint.
    (Point, Point): object_dispatch(point_point_dispatch_list),
    (Point, LineString): object_dispatch(point_linestring_dispatch_list),
    (Point, Polygon): object_dispatch(point_polygon_dispatch_list),
    (LineString, LineString): object_dispatch(
        linestring_linestring_dispatch_list
    ),
    (LineString, Polygon): object_dispatch(linestring_polygon_dispatch_list),
    (Polygon, Polygon): object_dispatch(polygon_polygon_dispatch_list),
}


def simple_test_dispatch():
    """Generates a list of test cases for each geometry type combination.

    Each dispatched test case is a tuple of the form:
    (test_name, test_description, lhs, rhs)
    which is run in `test_binpred_test_dispatch.py`.

    The test_name is a unique identifier for the test case.
    The test_description is a string representation of the test case.
    The lhs and rhs are GeoSeries of the left and right geometries.

    lhs and rhs are always constructed as a list of 3 geometries since
    the binpred function is designed to operate primarily on groups of
    geometries. The first and third feature in the list always match
    the first geometry specified in `test_description`, and the rhs is always
    a group of three of the second geometry specified in `test_description`.
    The second feature in the lhs varies.

    When the types of the lhs and rhs are equal, the second geometry
    from `test_description` is substituted for the second geometry in the lhs.
    This produces a test form of:
    lhs     rhs
    A       B
    B       B
    A       B

    This decision has two primary benefits:
    1. It causes the test to produce varied results (meaning results of the
    form (True, False, True) or (False, True, False), greatly reducing the
    likelihood of an "all-False" or "all-True" predicate producing
    false-positive results.
    2. It tests every binary predicate against self, such as A.touches(A)
    for every predicate and geometry combination.

    When the types of lhs and rhs are not equal this variation is not
    performed, since we cannot currently use predicate operations on mixed
    geometry types.
    """
    for types in type_dispatch:
        generator = type_dispatch[types]
        for test_name, test_description, lhs, rhs in generator:
            yield (
                test_name,
                test_description,
                cuspatial.GeoSeries(
                    [
                        lhs,
                        rhs if types[0] == types[1] else lhs,
                        lhs,
                    ]
                ),
                cuspatial.GeoSeries(
                    [
                        rhs,
                        rhs,
                        rhs,
                    ]
                ),
            )


@pytest.fixture(params=simple_test_dispatch())
def simple_test(request):
    """Generates a unique test case for each geometry type combination."""
    return request.param
