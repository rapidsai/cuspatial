# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_intersects import (
    IntersectsPredicateBase,
    PointLineStringIntersects,
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
        and then negate the result.

        Used by:
        (Point, Point)
        (Point, Polygon)
        (Polygon, Point)
        """
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


class LineStringLineStringDisjoint(IntersectsPredicateBase):
    def _postprocess(self, lhs, rhs, op_result):
        """Disjoint is the opposite of intersects, so just implement intersects
        and then negate the result."""
        result = super()._postprocess(lhs, rhs, op_result)
        return ~result


DispatchDict = {
    (Point, Point): ContainsDisjoint,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): PointLineStringDisjoint,
    (Point, Polygon): ContainsDisjoint,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointDisjoint,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringDisjoint,
    (LineString, Polygon): NotImplementedPredicate,
    (Polygon, Point): ContainsDisjoint,
    (Polygon, MultiPoint): NotImplementedPredicate,
    (Polygon, LineString): NotImplementedPredicate,
    (Polygon, Polygon): NotImplementedPredicate,
}
