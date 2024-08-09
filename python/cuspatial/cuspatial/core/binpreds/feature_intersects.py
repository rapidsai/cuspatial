# Copyright (c) 2023-2024, NVIDIA CORPORATION.


import cupy as cp

import cudf

from cuspatial.core.binops.intersection import pairwise_linestring_intersection
from cuspatial.core.binpreds.basic_predicates import (
    _basic_contains_any,
    _basic_intersects,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    IntersectsOpResult,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.feature_equals import EqualsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)


class IntersectsPredicateBase(BinPred):
    """Base class for binary predicates that are defined in terms of
    the intersection primitive.
    """

    def _preprocess(self, lhs, rhs):
        """Convert input lhs and rhs into LineStrings.

        The intersection basic predicate requires that the input
        geometries be LineStrings or MultiLineStrings. This function
        converts the input geometries into LineStrings and stores them
        in the PreprocessorResult object that is passed into
        _compute_predicate.

        Parameters
        ----------
        lhs : GeoSeries
            The left hand side of the binary predicate.
        rhs : GeoSeries
            The right hand side of the binary predicate.

        Returns
        -------
        GeoSeries
            A boolean Series containing the computed result of the
            final binary predicate operation,
        """
        return self._compute_predicate(lhs, rhs, PreprocessorResult(lhs, rhs))

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        """Compute the predicate using the intersects basic predicate.
        lhs and rhs must both be LineStrings or MultiLineStrings.
        """
        basic_result = pairwise_linestring_intersection(
            preprocessor_result.lhs, preprocessor_result.rhs
        )
        computed_result = IntersectsOpResult(basic_result)
        return self._postprocess(lhs, rhs, computed_result)

    def _get_intersecting_geometry_indices(self, lhs, op_result):
        """Naively computes the indices of matches by constructing
        a set of lengths from the returned offsets buffer, then
        returns an integer index for all of the offset sizes that
        are larger than 0."""
        is_offsets = cudf.Series._from_column(op_result.result[0])
        is_sizes = is_offsets[1:].reset_index(drop=True) - is_offsets[
            :-1
        ].reset_index(drop=True)
        return cp.arange(len(lhs))[is_sizes > 0]

    def _postprocess(self, lhs, rhs, op_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        match_indices = self._get_intersecting_geometry_indices(lhs, op_result)
        result = _false_series(len(lhs))
        if len(op_result.result[1]) > 0:
            result[match_indices] = True
        return result


class IntersectsByEquals(EqualsPredicateBase):
    pass


class PolygonPointIntersects(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_contains_any(lhs, rhs)


class PointPolygonIntersects(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_contains_any(rhs, lhs)


class LineStringPointIntersects(IntersectsPredicateBase):
    def _preprocess(self, lhs, rhs):
        return _basic_intersects(lhs, rhs)


class PointLineStringIntersects(LineStringPointIntersects):
    def _preprocess(self, lhs, rhs):
        """Swap LHS and RHS and call the normal contains processing."""
        return super()._preprocess(rhs, lhs)


class LineStringPolygonIntersects(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_contains_any(rhs, lhs)


class PolygonLineStringIntersects(BinPred):
    def _preprocess(self, lhs, rhs):
        return _basic_contains_any(lhs, rhs)


class PolygonPolygonIntersects(BinPred):
    def _preprocess(self, lhs, rhs):
        contains_rhs = _basic_contains_any(rhs, lhs)
        contains_lhs = _basic_contains_any(lhs, rhs)

        return contains_rhs | contains_lhs


""" Type dispatch dictionary for intersects binary predicates. """
DispatchDict = {
    (Point, Point): IntersectsByEquals,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): PointLineStringIntersects,
    (Point, Polygon): PointPolygonIntersects,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointIntersects,
    (LineString, MultiPoint): LineStringPointIntersects,
    (LineString, LineString): IntersectsPredicateBase,
    (LineString, Polygon): LineStringPolygonIntersects,
    (Polygon, Point): PolygonPointIntersects,
    (Polygon, MultiPoint): NotImplementedPredicate,
    (Polygon, LineString): PolygonLineStringIntersects,
    (Polygon, Polygon): PolygonPolygonIntersects,
}
