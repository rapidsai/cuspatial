# Copyright (c) 2023, NVIDIA CORPORATION.

import geopandas
import numpy as np
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


def sample_test_data(features, dispatch_list, size, lib=cuspatial):
    """Creates either a cuspatial or geopandas GeoSeries object using the
    Feature objects in `features`, the list of features to sample from in
    `dispatch_list`, and the size of the resultant GeoSeries.
    """
    geometry_tuples = [features[key][1:3] for key in dispatch_list]
    geometries = [
        [lhs_geo for lhs_geo, _ in geometry_tuples],
        [rhs_geo for _, rhs_geo in geometry_tuples],
    ]
    lhs = lib.GeoSeries(list(geometries[0]))
    rhs = lib.GeoSeries(list(geometries[1]))
    lhs_picks = np.repeat(np.arange(len(lhs)), len(lhs))
    rhs_picks = np.tile(np.arange(len(rhs)), len(rhs))
    return (
        lhs[lhs_picks].reset_index(drop=True),
        rhs[rhs_picks].reset_index(drop=True),
    )


def run_test(pred, dispatch_list):
    size = 10000
    lhs, rhs = sample_test_data(features, dispatch_list, size, cuspatial)
    gpdlhs, gpdrhs = sample_test_data(features, dispatch_list, size, geopandas)

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


def test_point_point_large_examples(predicate):  # noqa: F811
    run_test(predicate, point_point_dispatch_list)


def test_point_linestring_large_examples(predicate):  # noqa: F811
    run_test(predicate, point_linestring_dispatch_list)


def test_point_polygon_large_examples(predicate):  # noqa: F811
    run_test(predicate, point_polygon_dispatch_list)


def test_linestring_linestring_large_examples(predicate):  # noqa: F811
    run_test(predicate, linestring_linestring_dispatch_list)


def test_linestring_polygon_large_examples(predicate):  # noqa: F811
    run_test(predicate, linestring_polygon_dispatch_list)


def test_polygon_polygon_large_examples(predicate):  # noqa: F811
    run_test(predicate, polygon_polygon_dispatch_list)
