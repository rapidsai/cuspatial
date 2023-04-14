# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import Generic, TypeVar, Union

import cupy as cp

import cudf
from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binops.intersection import pairwise_linestring_intersection
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    ContainsOpResult,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.core.binpreds.complex_geometry_predicate import ComplexGeometryPredicate
from cuspatial.core.binpreds.contains import contains
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _count_results_in_multipoint_geometries,
    _false_series,
    _linestrings_from_geometry,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
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
        breakpoint()
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
        # _compute predicate no longer cares about preprocessor result
        # because information is passed directly to the postprocessor.
        # Creates an op_result and passes it and the preprocessor result
        # to the postprocessor.

        # Calls various _basic_predicate methods to compute the
        # predicate.
        # .contains calls .basic_contains_properly and also .basic_intersects
        # in order to assemble boundary-exclusive contains with intersection
        # results.
        """Compute the contains_properly relationship between two GeoSeries.
        A feature A contains another feature B if no points of B lie in the
        exterior of A, and at least one point of the interior of B lies in the
        interior of A. This is the inverse of `within`."""
        if not contains_only_polygons(lhs):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )
        points = preprocessor_result.final_rhs
        point_indices = preprocessor_result.point_indices
        if self._should_use_quadtree(lhs):
            result = contains(lhs, points, how="quadtree")
        else:
            result = contains(lhs, points, how="byte-limited")
        op_result = ContainsOpResult(result, points, point_indices)
        return self._postprocess(lhs, rhs, None, op_result)

    def _convert_quadtree_result_from_part_to_polygon_indices(
        self, lhs, point_result
    ):
        """Convert the result of a quadtree contains_properly call from
        part indices to polygon indices.

        Parameters
        ----------
        point_result : cudf.Series
            The result of a quadtree contains_properly call. This result
            contains the `part_index` of the polygon that contains the
            point, not the polygon index.

        Returns
        -------
        cudf.Series
            The result of a quadtree contains_properly call. This result
            contains the `polygon_index` of the polygon that contains the
            point, not the part index.
        """
        # Get the length of each part, map it to indices, and store
        # the result in a dataframe.
        rings_to_parts = cp.array(lhs.polygons.part_offset)
        part_sizes = rings_to_parts[1:] - rings_to_parts[:-1]
        parts_map = cudf.Series(
            cp.arange(len(part_sizes)), name="part_index"
        ).repeat(part_sizes)
        parts_index_mapping_df = parts_map.reset_index(drop=True).reset_index()
        # Map the length of each polygon in a similar fashion, then
        # join them below.
        parts_to_geoms = cp.array(lhs.polygons.geometry_offset)
        geometry_sizes = parts_to_geoms[1:] - parts_to_geoms[:-1]
        geometry_map = cudf.Series(
            cp.arange(len(geometry_sizes)), name="polygon_index"
        ).repeat(geometry_sizes)
        geom_index_mapping_df = geometry_map.reset_index(drop=True)
        geom_index_mapping_df.index.name = "part_index"
        geom_index_mapping_df = geom_index_mapping_df.reset_index()
        # Replace the part index with the polygon index by join
        part_result = parts_index_mapping_df.merge(
            point_result, on="part_index"
        )
        # Replace the polygon index with the row index by join
        return geom_index_mapping_df.merge(part_result, on="part_index")[
            ["polygon_index", "point_index"]
        ]

    def _reindex_allpairs(self, lhs, op_result) -> Union[Series, DataFrame]:
        """Prepare the allpairs result of a contains_properly call as
        the first step of postprocessing.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand side of the binary predicate.
        op_result : ContainsProperlyOpResult
            The result of the contains_properly call.

        Returns
        -------
        cudf.DataFrame

        """
        # Convert the quadtree part indices df into a polygon indices df
        polygon_indices = (
            self._convert_quadtree_result_from_part_to_polygon_indices(
                lhs, op_result.pip_result
            )
        )
        # Because the quadtree contains_properly call returns a list of
        # points that are contained in each part, parts can be duplicated
        # once their index is converted to a polygon index.
        allpairs_result = polygon_indices.drop_duplicates()

        # Replace the polygon index with the original index
        allpairs_result["polygon_index"] = allpairs_result[
            "polygon_index"
        ].replace(Series(lhs.index, index=cp.arange(len(lhs.index))))

        return allpairs_result

    def _postprocess(self, lhs, rhs, preprocessor_result, op_result):
        # Downstream predicates inherit from ComplexGeometryPredicate
        # that implements
        # point reconstruction for complex types separately.
        # Early return if individual points are required for downstream
        # predicates. Handle `any`, `all`, `none` modes.
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
        # Postprocessing early termination. Basic requests, or allpairs
        # requests do not do object reconstruction.
        if self.config.allpairs:
            return allpairs_result
        elif self.config.mode == "basic_none":
            final_result = cudf.Series(cp.repeat([True], len(lhs)))
            final_result.loc[allpairs_result["point_index"]] = False
            return final_result
        elif self.config.mode == "basic_any":
            final_result = _false_series(len(op_result.point_indices))
            final_result.loc[allpairs_result["point_index"]] = True
            return final_result
        elif self.config.mode == "basic_all":
            sizes = op_result.point_indices[1:] - op_result.point_indices[:-1]
            result_sizes = allpairs_result["point_index"].value_counts()
            final_result = _false_series(len(op_result.point_indices))
            final_result.loc[sizes == result_sizes] = True
            return final_result

        if len(op_result.pip_result) == 0:
            return _false_series(len(lhs))

        # Convert the quadtree part indices df into a polygon indices df.
        # Helps with handling multipolygons.
        allpairs_result = self._reindex_allpairs(lhs, op_result)
        # for each input pair i: result[i] =  true iff point[i] is
        # contained in at least one polygon of multipolygon[i].
        # pairwise
        final_result = _false_series(len(rhs))
        if len(lhs) == len(rhs):
            matches = (
                allpairs_result["polygon_index"]
                == allpairs_result["point_index"]
            )
            polygon_indexes = allpairs_result["polygon_index"][matches]
            final_result.loc[
                op_result.point_indices[polygon_indexes]
            ] = True
            return final_result
        else:
            final_result.loc[allpairs_result["polygon_index"]] = True
            return final_result


class ContainsPredicate(ContainsPredicateBase):
    def _compute_results(self, lhs, rhs, preprocessor_result):
        # Compute the contains predicate for the given lhs and rhs.
        # lhs and rhs are both cudf.Series of shapely geometries.
        # Returns a ContainsOpResult object.
        return lhs._contains(rhs)


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): ContainsPredicateBase,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): ContainsPredicateBase,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): ContainsPredicateBase,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsPredicateBase,
    (Polygon, MultiPoint): ContainsPredicateBase,
    (Polygon, LineString): ContainsPredicateBase,
    (Polygon, Polygon): ContainsPredicateBase,
}
