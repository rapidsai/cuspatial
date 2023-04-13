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
    _false_series,
    _linestring_to_boundary,
    _multipoints_from_geometry,
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
        """A and B have at least one point in common, and the common points
        lie in at least one boundary"""
        lhs_boundary = _linestring_to_boundary(lhs)
        rhs_boundary = _linestring_to_boundary(rhs)
        point_intersections = lhs._basic_intersects_at_point_only(rhs)
        boundary_intersects = lhs_boundary._basic_intersects(rhs_boundary)
        return point_intersections & boundary_intersects


class LineStringPolygonTouches(BinPred):
    def _preprocess(self, lhs, rhs):
        lhs_boundary = _linestring_to_boundary(lhs)
        rhs_boundary = _polygon_to_boundary(rhs)
        boundary_intersects = lhs_boundary._basic_intersects(rhs_boundary)
        interior_contains_any = rhs._basic_contains_any(lhs)
        return boundary_intersects & ~interior_contains_any


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
