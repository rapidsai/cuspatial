# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import TypeVar

from cuspatial.core.binpreds.binpred_interface import (
    ImpossiblePredicate,
    NotImplementedPredicate,
)
from cuspatial.core.binpreds.complex_geometry_predicate import (
    ComplexGeometryPredicate,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _multipoints_from_geometry,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsPredicateBase(ComplexGeometryPredicate):
    def __init__(self, **kwargs):
        """`ContainsProperlyPredicateBase` constructor.

        Parameters
        ----------
        allpairs: bool
            Whether to compute all pairs of features in the left-hand and
            right-hand GeoSeries. If False, the feature will be compared in a
            1:1 fashion with the corresponding feature in the other GeoSeries.
        """
        super().__init__(**kwargs)
        self.config.allpairs = kwargs.get("allpairs", False)
        self.config.mode = kwargs.get("mode", "full")

    def _preprocess(self, lhs, rhs):
        # Preprocess multi-geometries and complex geometries into
        # the correct input type for the contains predicate.
        # This is done by saving the shapes of multi-geometries,
        # then converting them all to single geometries.
        # Single geometries are converted from their original
        # lhs and rhs types to the types needed for the contains predicate.

        # point_indices: the indices of the points in the original
        # geometry.
        # geometry_offsets: the offsets of the multi-geometries in
        # the original geometry.
        preprocessor_result = super()._preprocess_multi(lhs, rhs)
        return self._compute_predicate(lhs, rhs, preprocessor_result)

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        contains = lhs._basic_contains_count(rhs).reset_index(drop=True)
        rhs_points = _multipoints_from_geometry(rhs)
        intersects = lhs._basic_intersects_count(rhs_points).reset_index(
            drop=True
        )
        # TODO: Need to handle multipolygon case. The [0] below ignores all
        # but the first polygon in a multipolygon.
        # TODO: Need better point counting in intersection.
        return contains + intersects >= rhs.sizes


class ContainsPredicate(ContainsPredicateBase):
    def _compute_results(self, lhs, rhs, preprocessor_result):
        # Compute the contains predicate for the given lhs and rhs.
        # lhs and rhs are both cudf.Series of shapely geometries.
        # Returns a ContainsOpResult object.
        return lhs._contains(rhs)


class PointPointContains(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        return lhs._basic_equals(rhs)


class LineStringMultiPointContainsPredicate(ContainsPredicateBase):
    def _compute_results(self, lhs, rhs, preprocessor_result):
        # Compute the contains predicate for the given lhs and rhs.
        # lhs and rhs are both cudf.Series of shapely geometries.
        # Returns a ContainsOpResult object.
        return lhs._linestring_multipoint_contains(rhs)


class LineStringLineStringContainsPredicate(ContainsPredicateBase):
    def _preprocess(self, lhs, rhs):
        count = lhs._basic_equals_count(rhs)
        return count == rhs.sizes


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): PointPointContains,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): ContainsPredicateBase,
    (LineString, MultiPoint): LineStringMultiPointContainsPredicate,
    (LineString, LineString): LineStringLineStringContainsPredicate,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsPredicateBase,
    (Polygon, MultiPoint): ContainsPredicateBase,
    (Polygon, LineString): ContainsPredicateBase,
    (Polygon, Polygon): ContainsPredicateBase,
}
