# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import Generic, TypeVar

import cupy as cp

import cudf
from cudf.core.series import Series

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedRoot,
)
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.core.binpreds.feature_equals import (
    DispatchDict as EQUALS_DISPATCH_DICT,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _count_results_in_multipoint_geometries,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
)

GeoSeries = TypeVar("GeoSeries")


class RootContains(BinPred, Generic[GeoSeries]):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Contains
    predicate is defined in terms of a Point-Point Intersects predicate.
    """

    def __init__(self, **kwargs):
        self.align = kwargs.get("align", False)
        self.allpairs = kwargs.get("allpairs", False)

    def _preprocess(self, lhs, rhs):
        """Flatten any rhs into only its points xy array. This is necessary
        because the root-level binary predicate only accepts points.

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
            the right-hand GeoSeries satisfies the requirements of a binary
            predicate with its corresponding feature in the left-hand
            GeoSeries.
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
        return self._op(lhs, final_rhs, Series(point_indices))

    def _should_use_quadtree(self):
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
        return (
            len(self.lhs) >= 32 or has_multipolygons(self.lhs) or self.allpairs
        )

    def _op(self, lhs, points, point_indices):
        """Compute the contains_properly relationship between two GeoSeries.
        A feature A contains another feature B if no points of B lie in the
        exterior of A, and at least one point of the interior of B lies in the
        interior of A. This is the inverse of `within`."""
        if not contains_only_polygons(lhs):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )
        if self._should_use_quadtree():
            result = contains_properly(lhs, points, how="quadtree")
        else:
            result = contains_properly(lhs, points, how="byte-limited")
        return self._postprocess(lhs, points, point_indices, result)

    def _convert_quadtree_result_from_part_to_polygon_indices(
        self, point_result
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
        rings_to_parts = cp.array(self.lhs.polygons.part_offset)
        part_sizes = rings_to_parts[1:] - rings_to_parts[:-1]
        parts_map = cudf.Series(
            cp.arange(len(part_sizes)), name="part_index"
        ).repeat(part_sizes)
        parts_index_mapping_df = parts_map.reset_index(drop=True).reset_index()
        # Map the length of each polygon in a similar fashion, then
        # join them below.
        parts_to_geoms = cp.array(self.lhs.polygons.geometry_offset)
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

    def _postprocess_quadtree_result(self, point_indices, point_result):
        """Postprocess the result of a quadtree contains_properly call.

        Parameters
        ----------
        point_indices : cudf.Series
            The indices of the points in the rhs GeoSeries.
        point_result : cudf.Series
            The result of a quadtree contains_properly call. This result
            contains the `part_index` of the polygon that contains the
            point, not the polygon index.

        Returns
        -------
        cudf.Series
            A Series of boolean values indicating whether each feature in
            the rhs GeoSeries is contained in the lhs GeoSeries.
        """
        if len(point_result) == 0:
            return cudf.Series([False] * len(self.lhs))

        # Convert the quadtree part indices df into a polygon indices df
        polygon_indices = (
            self._convert_quadtree_result_from_part_to_polygon_indices(
                point_result
            )
        )
        # Because the quadtree contains_properly call returns a list of
        # points that are contained in each part, parts can be duplicated
        # once their index is converted to a polygon index.
        allpairs_result = polygon_indices.drop_duplicates()

        # Replace the polygon index with the original index
        allpairs_result["polygon_index"] = allpairs_result[
            "polygon_index"
        ].replace(Series(self.lhs.index, index=cp.arange(len(self.lhs.index))))

        # If the user wants all pairs, return the result. Otherwise,
        # return a boolean series indicating whether each point is
        # contained in the corresponding polygon.
        if self.allpairs:
            return allpairs_result
        else:
            # for each input pair i: result[i] =  true iff point[i] is
            # contained in at least one polygon of multipolygon[i].
            # pairwise
            if len(self.lhs) == len(self.rhs):
                matches = (
                    allpairs_result["polygon_index"]
                    == allpairs_result["point_index"]
                )
                final_result = Series([False] * len(point_indices))
                final_result.loc[
                    allpairs_result["polygon_index"][matches]
                ] = True
                return final_result
            else:
                final_result = Series([False] * len(point_indices))
                final_result.loc[allpairs_result["polygon_index"]] = True
                return final_result

    def _postprocess(self, lhs, rhs, point_indices, op_result):
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
        """
        return self._postprocess_quadtree_result(point_indices, op_result)


class PolygonComplexContains(RootContains):
    """Base class for contains operations that use a complex object on
    the right hand side.

    This class is shared by the Polygon*Contains classes that use
    a non-points object on the right hand side: MultiPoint, LineString,
    MultiLineString, Polygon, and MultiPolygon."""

    def _postprocess(self, lhs, rhs, point_indices, allpairs_result):
        # for each input pair i: result[i] =  true iff point[i] is
        # contained in at least one polygon of multipolygon[i].
        # pairwise
        (hits, expected_count,) = _count_results_in_multipoint_geometries(
            point_indices, allpairs_result
        )
        result_df = hits.reset_index().merge(
            expected_count.reset_index(), on="rhs_index"
        )
        result_df["feature_in_polygon"] = (
            result_df["point_index_x"] >= result_df["point_index_y"]
        )
        final_result = cudf.Series(
            [False] * (point_indices.max().item() + 1)
        )  # point_indices is zero index
        final_result.loc[
            result_df["rhs_index"][result_df["feature_in_polygon"]]
        ] = True
        return final_result


class PointPointContains(RootContains):
    def _preprocess(self, lhs, rhs):
        """PointPointContains that simply calls the equals predicate on the
        points."""
        predicate = EQUALS_DISPATCH_DICT[(lhs.column_type, rhs.column_type)](
            align=self.align
        )
        return predicate(lhs, rhs)


"""DispatchDict listing the classes to use for each combination of
    left and right hand side types. """
DispatchDict = {
    (Point, Point): PointPointContains,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): NotImplementedRoot,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): NotImplementedRoot,
    (Polygon, Point): RootContains,
    (Polygon, MultiPoint): RootContains,
    (Polygon, LineString): RootContains,
    (Polygon, Polygon): RootContains,
}
