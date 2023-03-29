# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedRoot,
)
from cuspatial.core.binpreds.feature_intersects import (
    PointLineStringIntersects,
    RootIntersects,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class ContainsDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        """Disjoint is the opposite of contains, so just implement contains
        and then negate the result."""
        from cuspatial.core.binpreds.binpred_dispatch import CONTAINS_DISPATCH

        predicate = CONTAINS_DISPATCH[(lhs.column_type, rhs.column_type)](
            align=self.config.align
        )
        return ~predicate(lhs, rhs)


class PointLineStringDisjoint(PointLineStringIntersects):
    def _postprocess(self, lhs, rhs, op_result):
        """Disjoint is the opposite of intersects, so just implement intersects
        and then negate the result."""
        result = super()._postprocess(lhs, rhs, op_result)
        return ~result


class LineStringPointDisjoint(PointLineStringDisjoint):
    def _preprocess(self, lhs, rhs):
        """Swap ordering for Intersects."""
        return super()._preprocess(rhs, lhs)


class LineStringLineStringDisjoint(RootIntersects):
    def _postprocess(self, lhs, rhs, op_result):
        """Disjoint is the opposite of intersects, so just implement intersects
        and then negate the result."""
        result = super()._postprocess(lhs, rhs, op_result)
        return ~result


DispatchDict = {
    (Point, Point): ContainsDisjoint,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): PointLineStringDisjoint,
    (Point, Polygon): ContainsDisjoint,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): LineStringPointDisjoint,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): LineStringLineStringDisjoint,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): ContainsDisjoint,
    (Polygon, MultiPoint): ContainsDisjoint,
    (Polygon, LineString): ContainsDisjoint,
    (Polygon, Polygon): ContainsDisjoint,
}
