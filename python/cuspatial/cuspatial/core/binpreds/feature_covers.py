# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_any,
    _basic_contains_count,
    _basic_equals_count,
    _basic_intersects_pli,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import (
    LineStringPointIntersects,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _points_and_lines_to_multipoints,
    _zero_series,
)


class CoversPredicateBase(EqualsPredicateBase):
    """Implements the covers predicate across different combinations of
    geometry types.  For example, a Point-Polygon covers predicate is
    defined in terms of a Point-Polygon equals predicate.

    Point.covers(Point)
    LineString.covers(Polygon)
    """

    pass


class MultiPointMultiPointCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        # A multipoint A covers another multipoint B iff
        # every point in B is in A.
        # Count the number of points from rhs in lhs
        return lhs.contains(rhs)


class LineStringLineStringCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        # A linestring A covers another linestring B iff
        # no point in B is outside of A.
        return lhs.contains(rhs)


class PolygonPointCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_contains_any(lhs, rhs)


class PolygonLineStringCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        # A polygon covers a linestring if all of the points in the linestring
        # are in the interior or exterior of the polygon. This differs from
        # a polygon that contains a linestring in that some point of the
        # linestring must be in the interior of the polygon.
        # Count the number of points from rhs in the interior of lhs
        contains_count = _basic_contains_count(lhs, rhs)
        # Now count the number of points from rhs in the boundary of lhs
        pli = _basic_intersects_pli(lhs, rhs)
        intersections = pli[1]
        # There may be no intersection, so start with _zero_series
        equality = _zero_series(len(rhs))
        if len(intersections) > 0:
            matching_length_multipoints = _points_and_lines_to_multipoints(
                intersections, pli[0]
            )
            equality = _basic_equals_count(matching_length_multipoints, rhs)
        covers = contains_count + equality >= rhs.sizes
        return covers


class PolygonPolygonCovers(BinPred):
    def _preprocess(self, lhs, rhs):
        contains = lhs.contains(rhs)
        return contains


DispatchDict = {
    (Point, Point): CoversPredicateBase,
    (Point, MultiPoint): MultiPointMultiPointCovers,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): MultiPointMultiPointCovers,
    (MultiPoint, MultiPoint): MultiPointMultiPointCovers,
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
