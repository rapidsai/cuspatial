# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import (
    IntersectsPredicateBase,
    LineStringPointIntersects,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _linestrings_is_degenerate,
    _points_and_lines_to_multipoints,
    _zero_series,
)


class CoversPredicateBase(EqualsPredicateBase):
    """Implements the covers predicate across different combinations of
    geometry types.  For example, a Point-Polygon covers predicate is
    defined in terms of a Point-Point equals predicate. The initial release
    implements covers predicates that depend only on the equals predicate, or
    depend on no predicate, such as impossible cases like
    `LineString.covers(Polygon)`.

    For this initial release, cover is supported for the following types:

    Point.covers(Point)
    Point.covers(Polygon)
    LineString.covers(Polygon)
    Polygon.covers(Point)
    Polygon.covers(MultiPoint)
    Polygon.covers(LineString)
    Polygon.covers(Polygon)
    """

    pass


class LineStringLineStringCovers(IntersectsPredicateBase):
    def _preprocess(self, lhs, rhs):
        return rhs._basic_equals_all(lhs)


class PolygonPointCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        return lhs._basic_contains_any(rhs)


class PolygonLineStringCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        contains_count = lhs._basic_contains_count(rhs)
        pli = lhs._basic_intersects_pli(rhs)
        intersections = pli[1]
        equality = _zero_series(len(rhs))
        if len(intersections) == len(rhs):
            # If the result is degenerate
            is_degenerate = _linestrings_is_degenerate(intersections)
            # If all the points in the intersection are in the rhs
            equality = intersections._basic_equals_count(rhs)
            if len(is_degenerate) > 0:
                equality[is_degenerate] = 1
        elif len(intersections) > 0:
            matching_length_multipoints = _points_and_lines_to_multipoints(
                intersections, pli[0]
            )
            equality = matching_length_multipoints._basic_equals_count(rhs)
        return contains_count + equality >= rhs.sizes


class PolygonPolygonCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        contains = lhs.contains(rhs)
        return contains


DispatchDict = {
    (Point, Point): CoversPredicateBase,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringCovers,
    (LineString, Polygon): CoversPredicateBase,
    (Polygon, Point): PolygonPointCovers,
    (Polygon, MultiPoint): CoversPredicateBase,
    (Polygon, LineString): PolygonLineStringCovers,
    (Polygon, Polygon): PolygonPolygonCovers,
}
