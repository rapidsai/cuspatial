# Copyright (c) 2023, NVIDIA CORPORATION.

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils.column_utils import has_same_geometry


class RootOverlaps(RootEquals):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Crosses
    predicate is defined in terms of a Point-Point Crosses predicate.
    """

    def _preprocess(self, lhs, rhs):
        return self._op(lhs, rhs, rhs.point_indices)

    def _op(self, lhs, rhs, point_indices):
        result = super()._op(lhs, rhs, point_indices)
        return self._postprocess(lhs, rhs, point_indices, result)

    def _postprocess(self, lhs, rhs, point_indices, point_result):
        return super()._postprocess(lhs, rhs, point_indices, point_result)


class PointPointOverlaps(RootOverlaps):
    def _preprocess(self, lhs, rhs):
        """Points can't overlap other points, so we return False."""
        return cudf.Series(cp.tile(False, lhs.size))


class PolygonPointOverlaps(RootContains):
    def _postprocess(self, lhs, rhs, point_indices, point_result):
        if not has_same_geometry(self.lhs, self.rhs):
            return cudf.Series([False] * len(self.lhs))
        if len(point_result) == 0:
            return cudf.Series([False] * len(self.lhs))
        polygon_indices = (
            self._convert_quadtree_result_from_part_to_polygon_indices(
                point_result
            )
        )
        group_counts = polygon_indices.groupby("polygon_index").count()
        point_counts = (
            cudf.DataFrame(
                {"point_indices": point_indices, "input_size": True}
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
