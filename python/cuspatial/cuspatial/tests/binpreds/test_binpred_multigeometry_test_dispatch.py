# Copyright (c) 2023, NVIDIA CORPORATION.

import numpy as np
from binpred_test_dispatch import (  # noqa: F401
    features,
    linestring_linestring_dispatch_list,
    linestring_polygon_dispatch_list,
    point_linestring_dispatch_list,
    point_point_dispatch_list,
    point_polygon_dispatch_list,
    polygon_polygon_dispatch_list,
    predicate,
)

import cudf

import cuspatial


def sample_single_geometries(geometries):
    lhs = cuspatial.GeoSeries(list(geometries))
    singles = cudf.Series(np.arange(len(lhs)))
    multis = cudf.Series(np.repeat(np.arange(len(lhs)), len(lhs)))
    lhs_picks = cudf.concat([singles, multis]).reset_index(drop=True)
    return lhs[lhs_picks].reset_index(drop=True)


def sample_multi_geometries(geometries):
    rhs = cuspatial.GeoSeries(list(geometries))
    tiles = np.tile(np.arange(len(rhs)), len(rhs))
    repeats = np.repeat(np.arange(len(rhs)), len(rhs))
    singles = cudf.Series(np.arange(len(rhs)))
    multis = cudf.DataFrame(
        {"tile": tiles, "repeat": repeats}
    ).interleave_columns()
    rhs_picks = cudf.concat([singles, multis]).reset_index(drop=True)
    return rhs[rhs_picks].reset_index(drop=True)


def sample_cartesian_left(features, dispatch_list):
    """Returns each left geometry in `features` as a cuspatial GeoSeries object
    repeated the number of times in `dispatch_list`.
    """
    geometries = [features[key][1] for key in dispatch_list]
    return sample_single_geometries(geometries)


def sample_cartesian_right(features, dispatch_list):
    """Returns each right geometry in `features` as a cuspatial GeoSeries
    object tiled the number of times in `dispatch_list`.
    """
    geometries = [features[key][2] for key in dispatch_list]
    return sample_multi_geometries(geometries)


def sample_test_data(features, dispatch_list):
    """Creates either a cuSpatial or geopandas GeoSeries object using the
    features in `features` and the list of features to sample from
    `dispatch_list`.
    """
    geometry_tuples = [features[key][1:3] for key in dispatch_list]
    geometries = [
        [lhs_geo for lhs_geo, _ in geometry_tuples],
        [rhs_geo for _, rhs_geo in geometry_tuples],
    ]
    lhs = cuspatial.GeoSeries(list(geometries[0]))
    rhs = cuspatial.GeoSeries(list(geometries[1]))
    lhs_picks = np.repeat(np.arange(len(lhs)), len(lhs))
    rhs_picks = np.tile(np.arange(len(rhs)), len(rhs))
    return (
        lhs[lhs_picks].reset_index(drop=True),
        rhs[rhs_picks].reset_index(drop=True),
    )


def run_test(pred, lhs, rhs, gpdlhs, gpdrhs):
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


def test_point_multipoint(predicate):  # noqa: F811
    geometries = [features[key][2] for key in point_point_dispatch_list]
    points_lhs = sample_single_geometries(geometries)
    points_rhs = sample_cartesian_right(features, point_point_dispatch_list)
    size = len(point_point_dispatch_list)
    multipoint_offsets = np.concatenate(
        [np.arange(size), np.arange((size * size) + 1) * size + size]
    )
    multipoints_rhs = cuspatial.GeoSeries.from_multipoints_xy(
        points_rhs.points.xy, multipoint_offsets
    )
    gpdlhs = points_lhs.to_geopandas()
    gpdrhs = multipoints_rhs.to_geopandas()

    run_test(predicate, points_lhs, multipoints_rhs, gpdlhs, gpdrhs)


def test_multipoint_multipoint(predicate):  # noqa: F811
    points_lhs, points_rhs = sample_test_data(
        features, point_point_dispatch_list
    )
    lhs = cuspatial.GeoSeries.from_multipoints_xy(
        points_lhs.points.xy,
        np.arange(len(points_lhs)),
    )
    rhs = cuspatial.GeoSeries.from_multipoints_xy(
        points_rhs.points.xy, np.arange(len(points_rhs))
    )
    gpdlhs = lhs.to_geopandas()
    gpdrhs = rhs.to_geopandas()

    run_test(predicate, lhs, rhs, gpdlhs, gpdrhs)


def test_point_multilinestring(predicate):  # noqa: F811
    points_lhs = sample_cartesian_left(
        features, point_linestring_dispatch_list
    )
    lines_rhs = sample_cartesian_right(
        features, point_linestring_dispatch_list
    )
    size = len(point_linestring_dispatch_list)
    part_offset = np.concatenate([np.arange(len(lines_rhs) + 1) * 2])
    geometry_offset = np.concatenate(
        [np.arange(size), np.arange((size * size) + 1) * 2 + size]
    )
    multilinestrings_rhs = cuspatial.GeoSeries.from_linestrings_xy(
        lines_rhs.lines.xy, part_offset, geometry_offset
    )
    gpdlhs = points_lhs.to_geopandas()
    gpdrhs = multilinestrings_rhs.to_geopandas()

    run_test(predicate, points_lhs, multilinestrings_rhs, gpdlhs, gpdrhs)
