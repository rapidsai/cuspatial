# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import TypeVar

import cupy as cp

import cudf

from cuspatial.core.binpreds.basic_predicates import (
    _basic_equals_all,
    _basic_intersects,
)
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ContainsOpResult,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.core.binpreds.contains_geometry_processor import (
    ContainsGeometryProcessor,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _is_complex,
)
from cuspatial.utils.column_utils import (
    contains_only_polygons,
    has_multipolygons,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsProperlyPredicate(ContainsGeometryProcessor):
    def __init__(self, **kwargs):
        """Base class for binary predicates that are defined in terms of
        `contains_properly`.

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
        preprocessor_result = super()._preprocess_multipoint_rhs(lhs, rhs)
        return self._compute_predicate(lhs, rhs, preprocessor_result)

    def _pip_mode(self, lhs, rhs):
        """Determine if the quadtree should be used for the binary predicate.

        Returns
        -------
        bool
            True if the quadtree should be used, False otherwise.

        Notes
        -----
        1. If the number of polygons in the lhs is less than 32, we use the
           brute-force algorithm because it is faster and has less memory
           overhead.
        2. If the lhs contains multipolygons, or `allpairs=True` is specified,
           we use quadtree because the quadtree code path already handles
           multipolygons.
        3. Otherwise default to pairwise to match the default GeoPandas
           behavior.
        """
        if len(lhs) <= 31:
            return "brute_force"
        elif self.config.allpairs or has_multipolygons(lhs):
            return "quadtree"
        else:
            return "pairwise"

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
        mode = self._pip_mode(lhs, preprocessor_result.final_rhs)
        lhs_indices = lhs.index
        # Duplicates the lhs polygon for each point in the final_rhs result
        # that was computed by _preprocess. Will always ensure that the
        # number of points in the rhs is equal to the number of polygons in the
        # lhs.
        if mode == "pairwise":
            lhs_indices = preprocessor_result.point_indices
        pip_result = contains_properly(
            lhs[lhs_indices], preprocessor_result.final_rhs, mode=mode
        )
        # If the mode is pairwise or brute_force, we need to replace the
        # `pairwise_index` of each repeated polygon with the `part_index`
        # from the preprocessor result.
        if "pairwise_index" in pip_result.columns:
            pairwise_index_df = cudf.DataFrame(
                {
                    "pairwise_index": cp.arange(len(lhs_indices)),
                    "part_index": rhs.point_indices,
                }
            )
            pip_result = pip_result.merge(
                pairwise_index_df, on="pairwise_index"
            )[["part_index", "point_index"]]
        op_result = ContainsOpResult(pip_result, preprocessor_result)
        return self._postprocess(lhs, rhs, preprocessor_result, op_result)

    def _postprocess(self, lhs, rhs, preprocessor_result, op_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate.

        Postprocess for contains_properly has to handle multiple input and
        output configurations.


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

        if _is_complex(rhs):
            return super()._postprocess_multipoint_rhs(
                lhs, rhs, preprocessor_result, op_result, mode=self.config.mode
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
        return _basic_intersects(lhs, rhs)


class LineStringLineStringContainsProperly(BinPred):
    def _preprocess(self, lhs, rhs):
        count = _basic_equals_all(lhs, rhs)
        return count


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): ContainsProperlyByIntersection,
    (Point, MultiPoint): ContainsProperlyByIntersection,
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
