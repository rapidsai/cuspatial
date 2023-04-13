# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _linestring_to_boundary,
    _polygon_to_boundary,
)


class TouchesPredicateBase(ContainsPredicateBase):
    """Base class for binary predicates that use the contains predicate
    to implement the touches predicate. For example, a Point-Polygon
    Touches predicate is defined in terms of a Point-Polygon Contains
    predicate.

    Used by:
    (Point, Polygon)
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    (Polygon, Polygon)
    """

    def _compute_predicate(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        preprocessor_result: PreprocessorResult,
    ):
        # contains = lhs._basic_contains_any(rhs)
        equals = lhs._basic_equals(rhs)
        intersects = lhs._basic_intersects(rhs)
        return equals | intersects


class PointLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        return lhs._basic_equals(rhs)


class PointPolygonTouches(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        # Reverse argument order.
        equals_all = rhs._basic_equals_all(lhs)
        touches = rhs._basic_intersects(lhs)
        return ~equals_all & touches


class LineStringLineStringTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        boundary_touches = lhs._basic_equals(rhs)
        interior_intersects = lhs._basic_intersects(rhs)
        return boundary_touches & ~interior_intersects


class LineStringPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        breakpoint()
        lhs_boundary = _linestring_to_boundary(lhs)
        rhs_boundary = _polygon_to_boundary(rhs)
        boundary_touches = lhs_boundary._basic_equals(rhs_boundary)
        interior_intersects = lhs._basic_intersects(rhs)
        return boundary_touches & ~interior_intersects


DispatchDict = {
    (Point, Point): ImpossiblePredicate,
    (Point, MultiPoint): TouchesPredicateBase,
    (Point, LineString): PointLineStringTouches,
    (Point, Polygon): PointPolygonTouches,
    (MultiPoint, Point): TouchesPredicateBase,
    (MultiPoint, MultiPoint): TouchesPredicateBase,
    (MultiPoint, LineString): TouchesPredicateBase,
    (MultiPoint, Polygon): TouchesPredicateBase,
    (LineString, Point): TouchesPredicateBase,
    (LineString, MultiPoint): TouchesPredicateBase,
    (LineString, LineString): LineStringLineStringTouches,
    (LineString, Polygon): LineStringPolygonTouches,
    (Polygon, Point): TouchesPredicateBase,
    (Polygon, MultiPoint): TouchesPredicateBase,
    (Polygon, LineString): TouchesPredicateBase,
    (Polygon, Polygon): TouchesPredicateBase,
}
