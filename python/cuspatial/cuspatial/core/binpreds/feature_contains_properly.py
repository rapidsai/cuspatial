# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import Generic, TypeVar, Union

import cupy as cp

import cudf
from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ContainsOpResult,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.core.binpreds.feature_contains import ComplexGeometryPredicate, ContainsPredicateBase
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _count_results_in_multipoint_geometries,
    _false_series,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsProperlyPredicate(ContainsPredicateBase, ComplexGeometryPredicate):
    def __init__(self, **kwargs):
        """`ContainsProperlyPredicate` constructor.

        Parameters
        ----------
        allpairs: bool
            Whether to compute all pairs of features in the left-hand and
            right-hand GeoSeries. If False, the feature will be compared in a
            1:1 fashion with the corresponding feature in the other GeoSeries.
        """
        super().__init__(**kwargs)

    """Base class for binary predicates that are defined in terms of a
    `contains` basic predicate. This class implements the logic that underlies
    `polygon.contains` primarily, and is implemented for many cases.

    Subclasses are selected using the `DispatchDict` located at the end
    of this file.
    """
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        contains = super()._compute_predicate(lhs, rhs, preprocessor_result)
        intersects = lhs._basic_intersects_count(rhs)
        return self._postprocess(lhs, rhs, ContainsOpResult(
            lhs, rhs, preprocessor_result, contains, intersects)
        )

class ContainsProperlyByIntersection(BinPred):
    """Point types are contained only by an intersection test.

    Used by:
    (Point, Point)
    (LineString, Point)
    """

    def _preprocess(self, lhs, rhs):
        from cuspatial.core.binpreds.binpred_dispatch import (
            INTERSECTS_DISPATCH,
        )

        predicate = INTERSECTS_DISPATCH[(lhs.column_type, rhs.column_type)](
            align=self.config.align
        )
        return predicate(lhs, rhs)


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): ContainsProperlyByIntersection,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): ContainsProperlyByIntersection,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): ImpossiblePredicate,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsProperlyPredicate,
    (Polygon, MultiPoint): ContainsProperlyPredicate,
    (Polygon, LineString): ContainsProperlyPredicate,
    (Polygon, Polygon): ContainsProperlyPredicate,
}
