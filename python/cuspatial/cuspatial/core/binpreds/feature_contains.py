# Copyright (c) 2023, NVIDIA CORPORATION.

from abc import ABC, abstractmethod

import cupy as cp

from cudf.core.series import Series

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.binpred_interface import BinPred
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.core.geoseries import GeoSeries
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
)


class RootContains(BinPred):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Contains
    predicate is defined in terms of a Point-Point Intersects predicate.
    """

    @abstractmethod
    def __init__(self, lhs: GeoSeries, rhs: GeoSeries, **kwargs):
        super().__init__(lhs, rhs, **kwargs)
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self):
        return self._call()

    def _call(self):
        return self._preprocess()

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
            A GeoSeries of boolean values indicating whether each feature in the
            right-hand GeoSeries satisfies the requirements of a binary predicate
            with its corresponding feature in the left-hand GeoSeries.
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

    def _postprocess_quadtree_result(self, point_indices, point_result):
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
            if (
                contains_only_linestrings(self.rhs)
                or contains_only_polygons(self.rhs)
                or contains_only_multipoints(self.rhs)
            ):
                (
                    hits,
                    expected_count,
                ) = self._count_results_in_multipoint_geometries(
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
            else:
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


class PointPointContains(RootContains):
    pass


class PolygonPointContains(RootContains):
    pass


class PolygonMultiPointContains(RootContains):
    pass


class PolygonLineStringContains(RootContains):
    pass


class PolygonMultiLineStringContains(RootContains):
    pass


class PolygonPolygonContains(RootContains):
    pass


class PolygonMultiPolygonContains(RootContains):
    pass
