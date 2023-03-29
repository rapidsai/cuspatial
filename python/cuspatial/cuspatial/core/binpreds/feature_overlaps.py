# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf

from cuspatial.core.binpreds.binpred_interface import (
    ImpossibleRoot,
    NotImplementedRoot,
)
from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.core.binpreds.feature_equals import RootEquals
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)
from cuspatial.utils.column_utils import has_same_geometry


class RootOverlaps(RootEquals):
    """Base class for overlaps binary predicate. Depends on the
    equals predicate for all implementations up to this point.
    For example, a Point-Point Crosses predicate is defined in terms
    of a Point-Point Equals predicate.
    """

    pass


class PolygonPointOverlaps(RootContains):
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
    (Point, Point): ImpossibleRoot,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootOverlaps,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): ImpossibleRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): ImpossibleRoot,
    (LineString, Polygon): ImpossibleRoot,
    (Polygon, Point): RootOverlaps,
    (Polygon, MultiPoint): RootOverlaps,
    (Polygon, LineString): RootOverlaps,
    (Polygon, Polygon): RootOverlaps,
}
