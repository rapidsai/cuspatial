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
            "max_num_geometries": 10,
            "max_num_interior_rings": 10,
        },
    }

    ty0, ty1 = request.param
    g0, g1 = type_to_generator[ty0], type_to_generator[ty1]
    p0, p1 = generator_parameters[g0], generator_parameters[g1]
    return (partial(g0, **p0), partial(g1, **p1))


@pytest.mark.parametrize("n", [10, 1000])
@pytest.mark.parametrize("align", [True, False])
def test_geoseries_distance_non_empty(_binary_op_combinations, n, align):
    g0, g1 = _binary_op_combinations
    h0 = geopandas.GeoSeries([*g0(n=n)])
    h1 = geopandas.GeoSeries([*g1(n=n)], index=h0.index[::-1])

    expected = h0.distance(h1, align=align)

    d0 = cuspatial.GeoSeries(h0)
    d1 = cuspatial.GeoSeries(h1)

    actual = d0.distance(d1, align=align)

    assert_series_equal(actual, cudf.Series(expected))
