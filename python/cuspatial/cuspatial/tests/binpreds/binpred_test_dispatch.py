# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest
from shapely.geometry import LineString, Point, Polygon

import cuspatial
from cuspatial.testing.test_geometries import (  # noqa: F401
    features,
    linestring_linestring_dispatch_list,
    linestring_polygon_dispatch_list,
    point_linestring_dispatch_list,
    point_point_dispatch_list,
    point_polygon_dispatch_list,
    polygon_polygon_dispatch_list,
)

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
