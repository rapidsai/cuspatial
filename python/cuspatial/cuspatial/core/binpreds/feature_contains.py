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
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _count_results_in_multipoint_geometries,
    _false_series,
    _linestrings_from_polygons,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
)

GeoSeries = TypeVar("GeoSeries")


class ContainsPredicateBase(BinPred, Generic[GeoSeries]):
    """Base class for binary predicates that are defined in terms of a
    `contains` basic predicate. This class implements the logic that underlies
    `polygon.contains` primarily, and is implemented for many cases.

    Subclasses are selected using the `DispatchDict` located at the end
    of this file.
    """

    def __init__(self, **kwargs):
        """`ContainsPredicateBase` constructor.

        Parameters
        ----------
        allpairs: bool
            Whether to compute all pairs of features in the left-hand and
            right-hand GeoSeries. If False, the feature will be compared in a
            1:1 fashion with the corresponding feature in the other GeoSeries.
        """
        super().__init__(**kwargs)
        self.config.allpairs = kwargs.get("allpairs", False)

    def _preprocess(self, lhs, rhs):
        """Flatten any rhs into only its points xy array. This is necessary
        because the basic predicate for contains, point-in-polygon,
        only accepts points.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.

        Returns
        -------
        result : GeoSeries
            A GeoSeries of boolean values indicating whether each feature in
            the right-hand GeoSeries satisfies the requirements of the point-
            in-polygon basic predicate with its corresponding feature in the
            left-hand GeoSeries.
        """
        # RHS conditioning:
        point_indices = None
        # point in polygon
        if contains_only_linestrings(rhs):
            # condition for linestrings
            geom = rhs.lines
        elif contains_only_polygons(rhs) is True:
            # polygon in polygon
            geom = rhs.polygons
        elif contains_only_multipoints(rhs) is True:
            # mpoint in polygon
            geom = rhs.multipoints
        else:
            # no conditioning is required
            geom = rhs.points
        xy_points = geom.xy

        # Arrange into shape for calling point-in-polygon, intersection, or
        # equals
        point_indices = geom.point_indices()
        from cuspatial.core.geoseries import GeoSeries

        final_rhs = GeoSeries(GeoColumn._from_points_xy(xy_points._column))
        preprocess_result = PreprocessorResult(
            lhs, rhs, final_rhs, point_indices
        )
        return self._compute_predicate(lhs, rhs, preprocess_result)

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

    def _compute_basic_predicate(
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
        points = preprocessor_result.final_rhs
        point_indices = preprocessor_result.point_indices
        if self._should_use_quadtree(lhs):
            result = contains_properly(lhs, points, how="quadtree")
        else:
            result = contains_properly(lhs, points, how="byte-limited")
        op_result = ContainsOpResult(result, None, points, point_indices)
        return op_result

    def _compute_predicate(
        self,
        lhs: "GeoSeries",
        rhs: "GeoSeries",
        preprocessor_result: PreprocessorResult,
    ):
        op_result = self._compute_basic_predicate(
            lhs, rhs, preprocessor_result
        )
        return self._postprocess(lhs, rhs, op_result)

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
        op_result : ContainsOpResult
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

    def _postprocess(self, lhs, rhs, op_result):
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
        if len(op_result.pip_result) == 0:
            return _false_series(len(lhs))

        # Convert the quadtree part indices df into a polygon indices df.
        # Helps with handling multipolygons.
        allpairs_result = self._reindex_allpairs(lhs, op_result)

        # If the user wants all pairs, return the result. Otherwise,
        # return a boolean series indicating whether each point is
        # contained in the corresponding polygon.
        if self.config.allpairs:
            return allpairs_result
        else:
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


class PolygonComplexContains(ContainsPredicateBase):
    """Base class for contains operations that use a complex object on
    the right hand side.

    This class is shared by the Polygon*Contains classes that use
    a non-points object on the right hand side: MultiPoint, LineString,
    MultiLineString, Polygon, and MultiPolygon.

    Used by:
    (Polygon, MultiPoint)
    (Polygon, LineString)
    (Polygon, Polygon)
    """

    def _compute_intersects(self, lhs, rhs):
        ls_lhs = _linestrings_from_polygons(lhs)
        ls_rhs = _linestrings_from_polygons(rhs)
        basic_result = pairwise_linestring_intersection(ls_lhs, ls_rhs)
        return basic_result

    def _compute_predicate(self, lhs, rhs, preprocessor_output):
        pip_result = super()._compute_basic_predicate(
            lhs, rhs, preprocessor_output
        )
        intersects_result = self._compute_intersects(lhs, rhs)
        breakpoint()
        return self._postprocess(
            lhs,
            rhs,
            ContainsOpResult(
                pip_result,
                intersects_result,
                preprocessor_output.final_rhs,
                preprocessor_output.point_indices,
            ),
        )

    def _postprocess(self, lhs, rhs, op_result):
        # for each input pair i: result[i] =  true iff point[i] is
        # contained in at least one polygon of multipolygon[i].
        # pairwise
        point_indices = op_result.point_indices
        allpairs_result = self._reindex_allpairs(lhs, op_result.pip_result)

        if isinstance(allpairs_result, Series):
            return allpairs_result

        (hits, expected_count,) = _count_results_in_multipoint_geometries(
            point_indices, allpairs_result
        )
        result_df = hits.reset_index().merge(
            expected_count.reset_index(), on="rhs_index"
        )
        result_df["feature_in_polygon"] = (
            result_df["point_index_x"] >= result_df["point_index_y"]
        )
        final_result = _false_series(len(rhs))
        final_result.loc[
            result_df["rhs_index"][result_df["feature_in_polygon"]]
        ] = True

        offsets = cudf.Series(op_result.intersection_result[0])
        sizes = offsets[1:].reset_index(drop=True) - offsets[:-1].reset_index(
            drop=True
        )
        exterior_ring_offsets = lhs.polygons.ring_offset.take(
            lhs.polygons.geometry_offset
        )
        exterior_ring_sizes = (
            exterior_ring_offsets[1:] - exterior_ring_offsets[:-1]
        ) - 1
        intersection_size_matches = sizes == exterior_ring_sizes
        final_result[
            intersection_size_matches.index
        ] = intersection_size_matches
        return final_result


class ContainsByIntersection(BinPred):
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


class LineStringLineStringContains(BinPred):
    """LineString types are contained only by an equality test."""

    def _preprocess(self, lhs, rhs):
        from cuspatial.core.binpreds.binpred_dispatch import EQUALS_DISPATCH

        predicate = EQUALS_DISPATCH[(lhs.column_type, rhs.column_type)](
            align=self.config.align
        )
        return predicate(lhs, rhs)


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): ContainsByIntersection,
    (Point, MultiPoint): ImpossiblePredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): ImpossiblePredicate,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): NotImplementedPredicate,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): ContainsByIntersection,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringContains,
    (LineString, Polygon): ImpossiblePredicate,
    (Polygon, Point): ContainsPredicateBase,
    (Polygon, MultiPoint): PolygonComplexContains,
    (Polygon, LineString): PolygonComplexContains,
    (Polygon, Polygon): PolygonComplexContains,
}
