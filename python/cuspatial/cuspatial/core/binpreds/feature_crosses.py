# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.basic_predicates import (
    _basic_equals_count,
    _basic_intersects_count,
    _basic_intersects_pli,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import IntersectsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
    _lines_to_boundary_multipoints,
    _pli_lines_to_multipoints,
    _pli_points_to_multipoints,
)


class CrossesPredicateBase(EqualsPredicateBase):
    """Base class for binary predicates that are defined in terms of a
    the equals binary predicate. For example, a Point-Point Crosses
    predicate is defined in terms of a Point-Point Equals predicate.

    Used by:
    (Point, Polygon)
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    (Polygon, Polygon)
    """

    pass


class LineStringLineStringCrosses(IntersectsPredicateBase):
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        # A linestring crosses another linestring iff
        # they intersect, and none of the points of the
        # intersection are in the boundary of the other
        pli = _basic_intersects_pli(rhs, lhs)
        points = _pli_points_to_multipoints(pli)
        lines = _pli_lines_to_multipoints(pli)
        # Optimization: only compute the subsequent boundaries and equalities
        # of indexes that contain point intersections and do not contain line
        # intersections.
        lhs_boundary = _lines_to_boundary_multipoints(lhs)
        rhs_boundary = _lines_to_boundary_multipoints(rhs)
        lhs_boundary_matches = _basic_equals_count(points, lhs_boundary)
        rhs_boundary_matches = _basic_equals_count(points, rhs_boundary)
        lhs_crosses = lhs_boundary_matches != points.sizes
        rhs_crosses = rhs_boundary_matches != points.sizes
        crosses = (
            (points.sizes > 0)
            & (lhs_crosses & rhs_crosses)
            & (lines.sizes == 0)
        )
        return crosses


class LineStringPolygonCrosses(BinPred):
    def _preprocess(self, lhs, rhs):
        intersects = _basic_intersects_count(rhs, lhs) > 1
        touches = rhs.touches(lhs)
        contains = rhs.contains(lhs)
        return ~touches & intersects & ~contains


class PolygonLineStringCrosses(LineStringPolygonCrosses):
    def _preprocess(self, lhs, rhs):
        return super()._preprocess(rhs, lhs)


class PointPointCrosses(CrossesPredicateBase):
    def _preprocess(self, lhs, rhs):
        """Points can't cross other points, so we return False."""
        return _false_series(len(lhs))


DispatchDict = {
    (Point, Point): PointPointCrosses,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): CrossesPredicateBase,
    (MultiPoint, Point): ImpossiblePredicate,
    (MultiPoint, MultiPoint): ImpossiblePredicate,
    (MultiPoint, LineString): ImpossiblePredicate,
    (MultiPoint, Polygon): ImpossiblePredicate,
    (LineString, Point): ImpossiblePredicate,
    (LineString, MultiPoint): ImpossiblePredicate,
    (LineString, LineString): LineStringLineStringCrosses,
    (LineString, Polygon): LineStringPolygonCrosses,
    (Polygon, Point): CrossesPredicateBase,
    (Polygon, MultiPoint): CrossesPredicateBase,
    (Polygon, LineString): PolygonLineStringCrosses,
    (Polygon, Polygon): ImpossiblePredicate,
}
