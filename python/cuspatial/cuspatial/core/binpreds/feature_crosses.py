# Copyright (c) 2023, NVIDIA CORPORATION.

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import NotImplementedRoot
from cuspatial.core.binpreds.feature_equals import RootEquals


class RootCrosses(RootEquals):
    """Base class for binary predicates that are defined in terms of a
    the equals binary predicate. For example, a Point-Point Crosses
    predicate is defined in terms of a Point-Point Equals predicate.
    """

    def _preprocess(self, lhs, rhs):
        return self._compute_predicate(lhs, rhs, rhs.point_indices)

    def _compute_predicate(self, lhs, rhs, point_indices):
        result = super()._compute_predicate(lhs, rhs, point_indices)
        return self._postprocess(lhs, rhs, point_indices, result)

    def _postprocess(self, lhs, rhs, point_indices, point_result):
        return super()._postprocess(lhs, rhs, point_indices, point_result)


class PointPointCrosses(RootCrosses):
    def _preprocess(self, lhs, rhs):
        """Points can't cross other points, so we return False."""
        return cudf.Series(cp.tile(False, lhs.size))


Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON

DispatchDict = {
    (Point, Point): PointPointCrosses,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootCrosses,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): RootCrosses,
    (Polygon, MultiPoint): RootCrosses,
    (Polygon, LineString): RootCrosses,
    (Polygon, Polygon): RootCrosses,
}
