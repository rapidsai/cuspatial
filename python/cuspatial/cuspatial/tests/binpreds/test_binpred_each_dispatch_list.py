# Copyright (c) 2023, NVIDIA CORPORATION.

import geopandas
from binpred_test_dispatch import predicate  # noqa: F401

import cuspatial
from cuspatial.testing.test_geometries import (
    features,
    linestring_linestring_dispatch_list,
    linestring_polygon_dispatch_list,
    point_linestring_dispatch_list,
    point_point_dispatch_list,
    point_polygon_dispatch_list,
    polygon_polygon_dispatch_list,
)


def sample_test_data(features, dispatch_list, lib=cuspatial):
    """Creates either a cuSpatial or geopandas GeoSeries object using the
    features in `features` and the list of features to sample from
    `dispatch_list`.
    """
    geometry_tuples = [features[key][1:3] for key in dispatch_list]
    geometries = [
        [lhs_geo for lhs_geo, _ in geometry_tuples],
        [rhs_geo for _, rhs_geo in geometry_tuples],
    ]
    lhs = lib.GeoSeries(list(geometries[0]))
    rhs = lib.GeoSeries(list(geometries[1]))
    return (lhs, rhs)


def run_test(pred, dispatch_list):
    lhs, rhs = sample_test_data(features, dispatch_list, cuspatial)
    gpdlhs, gpdrhs = sample_test_data(features, dispatch_list, geopandas)

    # Reverse
    pred_fn = getattr(rhs, pred)
    got = pred_fn(lhs)
    gpd_pred_fn = getattr(gpdrhs, pred)
    expected = gpd_pred_fn(gpdlhs)
    assert (got.values_host == expected.values).all()

    # Forward
    pred_fn = getattr(lhs, pred)
    got = pred_fn(rhs)
    gpd_pred_fn = getattr(gpdlhs, pred)
    expected = gpd_pred_fn(gpdrhs)
    assert (got.values_host == expected.values).all()


def test_point_point_all_examples(predicate):  # noqa: F811
    run_test(predicate, point_point_dispatch_list)


def test_point_linestring_all_examples(predicate):  # noqa: F811
    run_test(predicate, point_linestring_dispatch_list)


def test_point_polygon_all_examples(predicate):  # noqa: F811
    run_test(predicate, point_polygon_dispatch_list)


def test_linestring_linestring_all_examples(predicate):  # noqa: F811
    run_test(predicate, linestring_linestring_dispatch_list)


def test_linestring_polygon_all_examples(predicate):  # noqa: F811
    run_test(predicate, linestring_polygon_dispatch_list)


def test_polygon_polygon_all_examples(predicate):  # noqa: F811
    run_test(predicate, polygon_polygon_dispatch_list)
