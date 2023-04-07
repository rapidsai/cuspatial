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
predicates. Each fixture returns a tuple of the form
(geotype, geotype, predicate, expected_result). The
geotype fixtures are used to generate the first two
elements of the tuple. The predicate fixture is used to
generate the third element of the tuple. The expected_result
fixture is used to generate the fourth element of the tuple.
"""

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


def get_feature(feature_type, feature_name):
    return feature_dispatch[feature_type][feature_name]


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


# Build a nested dictionary for the dispatch system
predicate_dispatch = {
    "single_equal": single_equal,
    "single_disjoint": single_disjoint,
    "triple_center_equal": triple_center_equal,
    "triple_center_disjoint": triple_center_disjoint,
    "border_overlap": border_overlap,
    "interior_overlap": interior_overlap,
}


def feature_test_dispatch(type0, type1, test_type):
    dispatch_function = predicate_dispatch[test_type]
    return dispatch_function(type0, type1)
