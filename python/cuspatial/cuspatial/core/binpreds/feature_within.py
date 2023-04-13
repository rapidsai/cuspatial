# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.feature_contains import (
    ContainsPredicateBase,
    PolygonComplexContains,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import IntersectsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _linestrings_from_geometry,
)


class WithinPredicateBase(EqualsPredicateBase):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Within
    predicate is defined in terms of a Point-Point Contains predicate.
    Used by:
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    """

    pass


class WithinIntersectsPredicate(IntersectsPredicateBase):
    def _preprocess(self, lhs, rhs):
        ls_lhs = _linestrings_from_geometry(lhs)
        ls_rhs = _linestrings_from_geometry(rhs)
        return self._compute_predicate(
            lhs, rhs, PreprocessorResult(ls_lhs, ls_rhs)
        )

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        intersects = rhs._basic_intersects(lhs)
        equals = rhs._basic_equals(lhs)
        return intersects & ~equals


class PointPointWithin(WithinPredicateBase):
    def _postprocess(self, lhs, rhs, op_result):
        return cudf.Series(op_result.result)


class PointLineStringWithin(WithinIntersectsPredicate):
    def _preprocess(self, lhs, rhs):
        # Note the order of arguments is reversed.
        return super()._preprocess(rhs, lhs)


class PointPolygonWithin(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        # Note the order of arguments is reversed.
        return super()._preprocess(rhs, lhs)


class LineStringLineStringWithin(IntersectsPredicateBase):
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        intersects = rhs._basic_intersects(lhs)
        equals = rhs._basic_equals_all(lhs)
        return intersects & equals


class ComplexPolygonWithin(PolygonComplexContains):
    """Implements within for complex polygons. Depends on contains result
    for the types.

    Used by:
    (MultiPoint, Polygon)
    (LineString, Polygon)
    (Polygon, Polygon)
    """

    def _preprocess(self, lhs, rhs):
        # Note the order of arguments is reversed.
        return super()._preprocess(rhs, lhs)


class LineStringPolygonWithin(BinPred):
    def _preprocess(self, lhs, rhs):
        contains_all = rhs._basic_contains_all(lhs)
        intersects = rhs._basic_intersects(lhs)
        contains_none = rhs._basic_contains_none(lhs)
        return contains_all & intersects


DispatchDict = {
    (Point, Point): PointPointWithin,
    (Point, MultiPoint): WithinIntersectsPredicate,
    (Point, LineString): PointLineStringWithin,
    (Point, Polygon): PointPolygonWithin,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): WithinIntersectsPredicate,
    (MultiPoint, Polygon): ComplexPolygonWithin,
    (LineString, Point): WithinIntersectsPredicate,
    (LineString, MultiPoint): WithinIntersectsPredicate,
    (LineString, LineString): LineStringLineStringWithin,
    (LineString, Polygon): LineStringPolygonWithin,
    (Polygon, Point): WithinPredicateBase,
    (Polygon, MultiPoint): WithinPredicateBase,
    (Polygon, LineString): WithinPredicateBase,
    (Polygon, Polygon): ComplexPolygonWithin,
}
