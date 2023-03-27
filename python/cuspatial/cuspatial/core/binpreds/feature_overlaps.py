# Copyright (c) 2023, NVIDIA CORPORATION.

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import (
    ContainsOpResult,
    NotImplementedRoot,
    PreprocessorResult,
)
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils.column_utils import has_same_geometry


class RootOverlaps(RootEquals):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Crosses
    predicate is defined in terms of a Point-Point Crosses predicate.
    """

    def _preprocess(self, lhs, rhs):
        return self._compute_predicate(
            lhs, rhs, PreprocessorResult(None, rhs.point_indices)
        )

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        result = super()._compute_predicate(lhs, rhs, preprocessor_result)
        return self._postprocess(
            lhs,
            rhs,
            ContainsOpResult(result, preprocessor_result.point_indices),
        )

    def _postprocess(self, lhs, rhs, op_result):
        return super()._postprocess(lhs, rhs, op_result)


class PointPointOverlaps(RootOverlaps):
    def _preprocess(self, lhs, rhs):
        """Points can't overlap other points, so we return False."""
        return cudf.Series(cp.tile(False, lhs.size))


class PolygonPointOverlaps(RootContains):
    def _postprocess(self, lhs, rhs, op_result):
        if not has_same_geometry(lhs, rhs):
            return cudf.Series([False] * len(lhs))
        if len(op_result.point_result) == 0:
            return cudf.Series([False] * len(lhs))
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


Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

DispatchDict = {
    (Point, Point): PointPointOverlaps,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootOverlaps,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): PolygonPointOverlaps,
    (Polygon, MultiPoint): RootOverlaps,
    (Polygon, LineString): PolygonPointOverlaps,
    (Polygon, Polygon): RootOverlaps,
}
