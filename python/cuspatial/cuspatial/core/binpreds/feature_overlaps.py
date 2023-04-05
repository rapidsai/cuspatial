# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf

from cuspatial.core.binpreds.binpred_interface import NotImplementedPredicate
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)
from cuspatial.utils.column_utils import has_same_geometry


class OverlapsPredicateBase(EqualsPredicateBase):
    """Base class for overlaps binary predicate. Depends on the
    equals predicate for all implementations up to this point.
    For example, a Point-Point Crosses predicate is defined in terms
    of a Point-Point Equals predicate.
    """

    pass


class PointPointOverlaps(OverlapsPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Points can't overlap other points, so we return False."""
        return _false_series(len(lhs))


class PolygonPointOverlaps(ContainsPredicateBase):
    def _postprocess(self, lhs, rhs, op_result):
        if not has_same_geometry(lhs, rhs) or len(op_result.point_result) == 0:
            return _false_series(len(lhs))
        polygon_indices = (
            self._convert_quadtree_result_from_part_to_polygon_indices(
                op_result.point_result
            )
        )
        group_counts = polygon_indices.groupby("polygon_index").count()
        point_counts = (
            cudf.DataFrame(
                {"point_indices": op_result.point_indices, "input_size": True}
            )
            .groupby("point_indices")
            .count()
        )
        result = (group_counts["point_index"] > 0) & (
            group_counts["point_index"] < point_counts["input_size"]
        )
        return result


"""Dispatch table for overlaps binary predicate."""
DispatchDict = {
    (Point, Point): PointPointOverlaps,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): NotImplementedPredicate,
    (Point, Polygon): OverlapsPredicateBase,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): NotImplementedPredicate,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): NotImplementedPredicate,
    (LineString, Polygon): NotImplementedPredicate,
    (Polygon, Point): OverlapsPredicateBase,
    (Polygon, MultiPoint): OverlapsPredicateBase,
    (Polygon, LineString): OverlapsPredicateBase,
    (Polygon, Polygon): OverlapsPredicateBase,
}
