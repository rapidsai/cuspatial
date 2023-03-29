# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedRoot,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
)


class RootDisjoint(BinPred):
    def _preprocess(self, lhs, rhs):
        """Disjoint is the opposite of contains, so just implement contains
        and then negate the result."""
        from cuspatial.core.binpreds.binpred_dispatch import CONTAINS_DISPATCH

        predicate = CONTAINS_DISPATCH[(lhs.column_type, rhs.column_type)](
            align=self.config.align
        )
        return ~predicate(lhs, rhs)


DispatchDict = {
    (Point, Point): RootDisjoint,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootDisjoint,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): RootDisjoint,
    (Polygon, MultiPoint): RootDisjoint,
    (Polygon, LineString): RootDisjoint,
    (Polygon, Polygon): RootDisjoint,
}
