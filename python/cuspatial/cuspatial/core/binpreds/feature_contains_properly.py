# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import TypeVar

import cupy as cp

import cudf

from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ContainsOpResult,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.core.binpreds.feature_contains import (
    ComplexGeometryPredicate,
    ContainsPredicateBase,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
    _is_complex,
)
from cuspatial.utils.column_utils import (
    contains_only_polygons,
    has_multipolygons,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsProperlyPredicate(
    ContainsPredicateBase, ComplexGeometryPredicate
):
    def __init__(self, **kwargs):
        """Base class for binary predicates that are defined in terms of a
        `contains` basic predicate. This class implements the logic that
        underlies `polygon.contains` primarily, and is implemented for many
        cases.

        Subclasses are selected using the `DispatchDict` located at the end
        of this file.

        Parameters
        ----------
        allpairs: bool
            Whether to compute all pairs of features in the left-hand and
            right-hand GeoSeries. If False, the feature will be compared in a
            1:1 fashion with the corresponding feature in the other GeoSeries.
        mode: str
            The mode to use for computing the predicate. The default is
            "full", which computes true or false if the `.contains_properly`
            predicate is satisfied. Other options include "basic_none",
            "basic_any", "basic_all", and "basic_count".
        """
        super().__init__(**kwargs)
        self.config.allpairs = kwargs.get("allpairs", False)
        self.config.mode = kwargs.get("mode", "full")

    def _preprocess(self, lhs, rhs):
        preprocessor_result = super()._preprocess_multi(lhs, rhs)
        return self._compute_predicate(lhs, rhs, preprocessor_result)

    def _should_use_quadtree(self, lhs):
        """Determine if the quadtree should be used for the binary predicate.

        Returns
        -------
        bool
            True if the quadtree should be used, False otherwise.

        Notes
        -----
        1. Quadtree is always used if user requests `allpairs=True`.
        2. If the number of polygons in the lhs is less than 32, we use the
           byte-limited algorithm because it is faster and has less memory
           overhead.
        3. If the lhs contains more than 32 polygons, we use the quadtree
           because it does not have a polygon-count limit.
        4. If the lhs contains multipolygons, we use quadtree because the
           performance between quadtree and byte-limited is similar, but
           code complexity would be higher if we did multipolygon
           reconstruction on both code paths.
        """
        return len(lhs) >= 32 or has_multipolygons(lhs) or self.config.allpairs

    def _compute_predicate(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        preprocessor_result: PreprocessorResult,
    ):
        """Compute the contains_properly relationship between two GeoSeries.
        A feature A contains another feature B if no points of B lie in the
        exterior of A, and at least one point of the interior of B lies in the
        interior of A. This is the inverse of `within`."""
        if not contains_only_polygons(lhs):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )
        if self._should_use_quadtree(lhs):
            pip_result = contains_properly(
                lhs, preprocessor_result.points, how="quadtree"
            )
        else:
            pip_result = contains_properly(
                lhs, preprocessor_result.points, how="byte-limited"
            )
        op_result = ContainsOpResult(pip_result, preprocessor_result)
        return self._postprocess(lhs, rhs, preprocessor_result, op_result)

    def _return_unprocessed_result(self, lhs, op_result, preprocessor_result):
        """Return the result of the basic predicate without any
        postprocessing.
        """
        reindex_pip_result = self._reindex_allpairs(lhs, op_result)
        if len(reindex_pip_result) == 0:
            if self.config.mode == "basic_count":
                return cudf.Series(cp.zeros(len(lhs), dtype="int32"))
            else:
                return _false_series(len(lhs))
        # Postprocessing early termination. Basic requests, or allpairs
        # requests do not do object reconstruction.
        if self.config.allpairs:
            return reindex_pip_result
        elif self.config.mode == "basic_none":
            final_result = cudf.Series(cp.repeat([True], len(lhs)))
            final_result.loc[reindex_pip_result["point_index"]] = False
            return final_result
        elif self.config.mode == "basic_any":
            final_result = _false_series(len(lhs))
            final_result.loc[reindex_pip_result["point_index"]] = True
            return final_result
        elif self.config.mode == "basic_all":
            sizes = (
                preprocessor_result.point_indices[1:]
                - preprocessor_result.point_indices[:-1]
            )
            result_sizes = reindex_pip_result["polygon_index"].value_counts()
            final_result = _false_series(
                len(preprocessor_result.point_indices)
            )
            final_result.loc[sizes == result_sizes] = True
            return final_result
        elif self.config.mode == "basic_count":
            return reindex_pip_result["polygon_index"].value_counts()

    def _postprocess(self, lhs, rhs, preprocessor_result, op_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate.

        Postprocess for contains_properly has to handle multiple input and
        output configurations.

        The input can be a single polygon, a single multipolygon, or a
        GeoSeries containing a mix of polygons and multipolygons.

        The input to postprocess is `point_indices`, which can be either a
        cudf.DataFrame with one row per point and one column per polygon or
        a cudf.DataFrame containing the point index and the part index for
        each point in the polygon.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand side of the binary predicate.
        rhs : GeoSeries
            The right-hand side of the binary predicate.
        preprocessor_output : ContainsOpResult
            The result of the contains_properly call.

        Returns
        -------
        cudf.Series or cudf.DataFrame
            A Series of boolean values indicating whether each feature in
            the rhs GeoSeries is contained in the lhs GeoSeries in the
            case of allpairs=False. Otherwise, a DataFrame containing the
            point index and the polygon index for each point in the
            polygon.
        """
        if self.config.mode != "full" or self.config.allpairs:
            return self._return_unprocessed_result(
                lhs, op_result, preprocessor_result
            )

        # for each input pair i: result[i] =  true iff point[i] is
        # contained in at least one polygon of multipolygon[i].
        if _is_complex(rhs):
            return super()._postprocess_multi(
                lhs, rhs, preprocessor_result, op_result
            )
        else:
            return super()._postprocess_points(
                lhs, rhs, preprocessor_result, op_result
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


class LineStringLineStringContainsProperly(BinPred):
    def _preprocess(self, lhs, rhs):
        count = lhs._basic_equals_all(rhs)
        return count


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
    (LineString, MultiPoint): ContainsProperlyPredicate,
    (LineString, LineString): LineStringLineStringContainsProperly,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsProperlyPredicate,
    (Polygon, MultiPoint): ContainsProperlyPredicate,
    (Polygon, LineString): ContainsProperlyPredicate,
    (Polygon, Polygon): ContainsProperlyPredicate,
}
