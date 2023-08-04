from functools import partial
from itertools import product

import geopandas
import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import cudf
from cudf.testing import assert_series_equal

import cuspatial

geometry_types = set(
    [Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon]
)


@pytest.fixture(params=set(product(geometry_types, geometry_types)))
def _binary_op_combinations(
    request,
    point_generator,
    multipoint_generator,
    linestring_generator,
    multilinestring_generator,
    polygon_generator,
    multipolygon_generator,
):
    type_to_generator = {
        Point: point_generator,
        MultiPoint: multipoint_generator,
        LineString: linestring_generator,
        MultiLineString: multilinestring_generator,
        Polygon: polygon_generator,
        MultiPolygon: multipolygon_generator,
    }

    generator_parameters = {
        point_generator: {},
        multipoint_generator: {"max_num_geometries": 10},
        linestring_generator: {"max_num_segments": 10},
        multilinestring_generator: {
            "max_num_geometries": 10,
            "max_num_segments": 10,
        },
        polygon_generator: {"distance_from_origin": 10, "radius": 10},
        multipolygon_generator: {
            "max_per_multi": 10,
            "distance_from_origin": 10,
            "radius": 10,
        },
    }

    ty0, ty1 = request.param
    g0, g1 = type_to_generator[ty0], type_to_generator[ty1]
    p0, p1 = generator_parameters[g0], generator_parameters[g1]
    return (partial(g0, **p0), partial(g1, **p1))


def test_geoseires_distance_empty():
    expected = geopandas.GeoSeries([]).distance(geopandas.GeoSeries([]))
    actual = cuspatial.GeoSeries([]).distance(cuspatial.GeoSeries([]))

    assert_series_equal(actual, cudf.Series(expected))


@pytest.mark.parametrize("n", [1, 100])
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize(
    "index_shuffle",
    [
        (lambda x: x, lambda x: reversed(x)),
        (lambda x: reversed(x), lambda x: x),
    ],
)
def test_geoseries_distance_non_empty(
    _binary_op_combinations, n, align, index_shuffle
):
    g0, g1 = _binary_op_combinations
    sfl0, sfl1 = index_shuffle

    h0 = geopandas.GeoSeries([*g0(n=n)], index=sfl0(range(n)))
    h1 = geopandas.GeoSeries([*g1(n=n)], index=sfl1(range(n)))

    expected = h0.distance(h1, align=align)

    d0 = cuspatial.GeoSeries(h0)
    d1 = cuspatial.GeoSeries(h1)

    actual = d0.distance(d1, align=align)

    assert_series_equal(actual, cudf.Series(expected))


def test_geoseries_distance_indices_different():
    h0 = geopandas.GeoSeries([Point(0, 0), Point(1, 1)], index=[0, 1])
    h1 = geopandas.GeoSeries([Point(0, 0)], index=[0])

    expected = h0.distance(h1)

    d0 = cuspatial.GeoSeries(h0)
    d1 = cuspatial.GeoSeries(h1)

    actual = d0.distance(d1)

    assert_series_equal(actual, cudf.Series(expected, nan_as_null=False))


def test_geoseries_distance_indices_different_not_aligned():
    h0 = geopandas.GeoSeries([Point(0, 0), Point(1, 1)], index=[0, 1])
    h1 = geopandas.GeoSeries([Point(0, 0)], index=[0])

    d0 = cuspatial.GeoSeries(h0)
    d1 = cuspatial.GeoSeries(h1)

    try:
        h0.distance(h1, align=False)
    except Exception as e:
        with pytest.raises(e.__class__, match=e.__str__()):
            d0.distance(d1, align=False)


@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize(
    "index",
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ],
)
def test_geoseries_distance_to_shapely_object(
    _binary_op_combinations, align, index
):
    g0, g1 = _binary_op_combinations
    h0 = geopandas.GeoSeries([*g0(n=10)], index=index)
    obj = [*g1(n=1)][0]

    expected = h0.distance(obj, align=align)

    d0 = cuspatial.GeoSeries(h0)

    actual = d0.distance(obj, align=align)

    assert_series_equal(actual, cudf.Series(expected))
