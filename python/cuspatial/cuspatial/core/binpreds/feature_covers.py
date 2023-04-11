# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binops.intersection import pairwise_linestring_intersection
from cuspatial.core.binpreds.binpred_interface import (
    IntersectsOpResult,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.feature_contains import ContainsPredicateBase
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.core.binpreds.feature_intersects import (
    IntersectsPredicateBase,
    LineStringPointIntersects,
    PointLineStringIntersects,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
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
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        """Compute the covers predicate using the intersects basic predicate.
        lhs and rhs must both be LineStrings or MultiLineStrings.
        """
        basic_result = pairwise_linestring_intersection(
            preprocessor_result.lhs, preprocessor_result.rhs
        )
        breakpoint()
        # TODO: Need to determine whether or not the intersection is a
        # linestring or a point.
        return self._postprocess(lhs, rhs, IntersectsOpResult(basic_result))

    def _postprocess(self, lhs, rhs, op_result):
        """Postprocess the result of the intersects operation."""
        if len(op_result.result[0]) - 1 != len(rhs):
            # The number of intersections is equal to the rhs side?
            # Where's my multi-processing here? I don't see this as a
            # column of values at all.
            return _false_series(lhs)
        else:
            return op_result.result[0]


class PolygonPolygonCovers(ContainsPredicateBase):
    pass


DispatchDict = {
    (Point, Point): CoversPredicateBase,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): PointLineStringIntersects,
    (Point, Polygon): CoversPredicateBase,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringCovers,
    (LineString, Polygon): CoversPredicateBase,
    (Polygon, Point): CoversPredicateBase,
    (Polygon, MultiPoint): CoversPredicateBase,
    (Polygon, LineString): CoversPredicateBase,
    (Polygon, Polygon): PolygonPolygonCovers,
}
