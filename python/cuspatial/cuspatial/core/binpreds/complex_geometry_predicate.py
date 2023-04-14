# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import Generic, TypeVar, Union

import cupy as cp

import cudf
from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binops.intersection import pairwise_linestring_intersection
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ContainsOpResult,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _count_results_in_multipoint_geometries,
    _false_series,
    _linestrings_from_geometry,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
)

GeoSeries = TypeVar("GeoSeries")


class ComplexFeaturePredicate(BinPred):
    def _preprocess_multi(self, lhs, rhs):
        # Breaks down complex geometries into their constituent parts.
        # Passes a tuple o the preprocessed geometries and a tuple of
        # the indices of the points in the original geometry.
        # This is used by the postprocessor to reconstruct the original
        # geometry.
        # Child classes should not implement this method.
        """Flatten any rhs into only its points xy array. This is necessary
        because the basic predicate for contains, point-in-polygon,
        only accepts points.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.

        Returns
        -------
        result : GeoSeries
            A GeoSeries of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of the point-
            in-polygon basic predicate with its corresponding feature in the
            left-hand GeoSeries.
        """
        # RHS conditioning:
        point_indices = None
        # point in polygon
        if contains_only_linestrings(rhs):
            # condition for linestrings
            geom = rhs.lines
        elif contains_only_polygons(rhs) is True:
            # polygon in polygon
            geom = rhs.polygons
        elif contains_only_multipoints(rhs) is True:
            # mpoint in polygon
            geom = rhs.multipoints
        else:
            # no conditioning is required
            geom = rhs.points
        xy_points = geom.xy

        # Arrange into shape for calling point-in-polygon, intersection, or
        # equals
        point_indices = geom.point_indices()
        from cuspatial.core.geoseries import GeoSeries

        final_rhs = GeoSeries(GeoColumn._from_points_xy(xy_points._column))
        preprocess_result = PreprocessorResult(
            lhs, rhs, final_rhs, point_indices
        )
        return preprocess_result

    def _postprocess_multi(self, lhs, rhs, preprocessor_result, op_result):
        # Doesn't use op_result, but uses preprocessor_result to
        # reconstruct the original geometry.
        # Child classes should call this method to reconstruct the
        # original geometry.

        # Complex geometry postprocessor
        point_indices = preprocessor_result.point_indices
        allpairs_result = self._reindex_allpairs(lhs, preprocessor_result)
        if isinstance(allpairs_result, Series):
            return allpairs_result

        (hits, expected_count,) = _count_results_in_multipoint_geometries(
            point_indices, allpairs_result
        )
        result_df = hits.reset_index().merge(
            expected_count.reset_index(), on="rhs_index"
        )
        result_df["feature_in_polygon"] = (
            result_df["point_index_x"] >= result_df["point_index_y"]
        )
        final_result = _false_series(len(rhs))
        final_result.loc[
            result_df["rhs_index"][result_df["feature_in_polygon"]]
        ] = True
        return final_result
