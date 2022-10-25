import geopandas as gpd
import pytest
from pandas.testing import assert_series_equal
from shapely.geometry import LineString, Point

import cuspatial


def test_empty():
    gs1 = cuspatial.GeoSeries([])
    gs2 = cuspatial.GeoSeries([])
    assert cuspatial.pairwise_point_distance(gs1, gs2).empty


def test_point_point_random(point_generator):
    gs1 = gpd.GeoSeries([*point_generator(100)])
    gs2 = gpd.GeoSeries([*point_generator(100)])

    cugs1 = cuspatial.from_geopandas(gs1)
    cugs2 = cuspatial.from_geopandas(gs2)

    got = cuspatial.pairwise_point_distance(cugs1, cugs2)
    expected = gs1.distance(gs2)
    assert_series_equal(got.to_pandas(), expected)


def test_point_multipoint_random(point_generator, multipoint_generator):
    gs1 = gpd.GeoSeries([*point_generator(100)])
    gs2 = gpd.GeoSeries([*multipoint_generator(100, 5)])

    cugs1 = cuspatial.from_geopandas(gs1)
    cugs2 = cuspatial.from_geopandas(gs2)

    got = cuspatial.pairwise_point_distance(cugs1, cugs2)
    expected = gs1.distance(gs2)
    assert_series_equal(got.to_pandas(), expected)


def test_multipoint_point_random(point_generator, multipoint_generator):
    gs1 = gpd.GeoSeries([*multipoint_generator(100, 5)])
    gs2 = gpd.GeoSeries([*point_generator(100)])

    cugs1 = cuspatial.from_geopandas(gs1)
    cugs2 = cuspatial.from_geopandas(gs2)

    got = cuspatial.pairwise_point_distance(cugs1, cugs2)
    expected = gs1.distance(gs2)
    assert_series_equal(got.to_pandas(), expected)


def test_multipoint_multipoint_random(multipoint_generator):
    gs1 = gpd.GeoSeries([*multipoint_generator(100, 5)])
    gs2 = gpd.GeoSeries([*multipoint_generator(100, 5)])

    cugs1 = cuspatial.from_geopandas(gs1)
    cugs2 = cuspatial.from_geopandas(gs2)

    got = cuspatial.pairwise_point_distance(cugs1, cugs2)
    expected = gs1.distance(gs2)
    assert_series_equal(got.to_pandas(), expected)


def test_mismatched_input_size():
    gs1 = cuspatial.GeoSeries([Point(0, 0)])
    gs2 = cuspatial.GeoSeries([Point(0, 0), Point(1, 1)])
    with pytest.raises(ValueError, match="must have the same length"):
        cuspatial.pairwise_point_distance(gs1, gs2)


def test_mismatched_input_type():
    gs1 = cuspatial.GeoSeries([Point(0, 0)])
    gs2 = cuspatial.GeoSeries([LineString([(0, 0), (1, 1)])])
    with pytest.raises(ValueError, match="must contain only points"):
        cuspatial.pairwise_point_distance(gs1, gs2)

    gs3 = cuspatial.GeoSeries([LineString([(0, 0), (1, 1)])])
    gs4 = cuspatial.GeoSeries([Point(0, 0)])
    with pytest.raises(ValueError, match="must contain only points"):
        cuspatial.pairwise_point_distance(gs3, gs4)
