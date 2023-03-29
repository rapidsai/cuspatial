# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf

from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils import binpred_utils
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class RootWithin(RootEquals):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Within
    predicate is defined in terms of a Point-Point Contains predicate.
    """

    pass


class PointPointWithin(RootWithin):
    def _postprocess(self, lhs, rhs, op_result):
        return cudf.Series(op_result.result)


class PointPolygonWithin(RootContains):
    def _preprocess(self, lhs, rhs):
        # Note the order of arguments is reversed.
        return super()._preprocess(rhs, lhs)


class ComplexPolygonWithin(RootContains):
    def _preprocess(self, lhs, rhs):
        # Note the order of arguments is reversed.
        return super()._preprocess(rhs, lhs)

    def _postprocess(self, lhs, rhs, op_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        (
            hits,
            expected_count,
        ) = binpred_utils._count_results_in_multipoint_geometries(
            op_result.point_indices, op_result.result
        )
        result_df = hits.reset_index().merge(
            expected_count.reset_index(), on="rhs_index"
        )
        result_df["feature_in_polygon"] = (
            result_df["point_index_x"] >= result_df["point_index_y"]
        )
        final_result = _false_series(len(lhs))
        final_result.loc[
            result_df["rhs_index"][result_df["feature_in_polygon"]]
        ] = True
        return final_result


DispatchDict = {
    (Point, Point): PointPointWithin,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): PointPolygonWithin,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): ComplexPolygonWithin,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): ComplexPolygonWithin,
    (Polygon, Point): RootWithin,
    (Polygon, MultiPoint): RootWithin,
    (Polygon, LineString): RootWithin,
    (Polygon, Polygon): ComplexPolygonWithin,
}
