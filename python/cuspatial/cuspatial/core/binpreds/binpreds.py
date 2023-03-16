# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from abc import ABC, abstractmethod

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.contains import (
    byte_limited_contains_properly,
    quadtree_contains_properly,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_multipolygons,
    has_same_geometry,
)


class BinaryPredicate(ABC):
    @abstractmethod
    def preprocess(self, lhs, rhs):
        """Preprocess the input data for the binary predicate. This method
        should be implemented by subclasses. Preprocess and postprocess are
        used to implement the discrete math rules of the binary predicates.

        Notes
        -----
        Should only be called once.

        Parameters
        ----------
        op : str
            The binary predicate to perform.
        lhs : GeoSeries
            The left-hand-side of the GeoSeries-level binary predicate.
        rhs : GeoSeries
            The right-hand-side of the GeoSeries-level binary predicate.

        Returns
        -------
        GeoSeries
            The left-hand-side of the internal binary predicate, may be
            reordered.
        GeoSeries
            The right-hand-side of the internal binary predicate, may be
            reordered.
        """
        pass

    @abstractmethod
    def postprocess(self, point_indices, point_result):
        """Postprocess the output data for the binary predicate. This method
        should be implemented by subclasses.

        Postprocess converts the raw results of the binary predicate into
        the final result. This is where the discrete math rules are applied.

        Parameters
        ----------
        op : str
            The binary predicate to post process. Determines for example the
            set predicate to use for computing the result.
        point_indices : cudf.Series
            The indices of the points in the original GeoSeries.
        point_result : cudf.Series
            The raw result of the binary predicate.

        Returns
        -------
        cudf.Series
            The output of the post processing, True/False results for
            the specified binary op.
        """
        pass

    def __init__(self, lhs, rhs, align=True):
        """Compute the binary predicate `op` on `lhs` and `rhs`.

        There are ten binary predicates supported by cuspatial:
        - `.equals`
        - `.disjoint`
        - `.touches`
        - `.contains`
        - `.contains_properly`
        - `.covers`
        - `.intersects`
        - `.within`
        - `.crosses`
        - `.overlaps`

        There are thirty-six ordering combinations of `lhs` and `rhs`, the
        unordered pairs of each `point`, `multipoint`, `linestring`,
        `multilinestring`, `polygon`, and `multipolygon`. The ordering of
        `lhs` and `rhs` is important because the result of the binary
        predicate is not symmetric. For example, `A.contains(B)` is not
        the same as `B.contains(A)`.

        Parameters
        ----------
        op : str
            The binary predicate to perform.
        lhs : GeoSeries
            The left-hand-side of the binary predicate.
        rhs : GeoSeries
            The right-hand-side of the binary predicate.
        align : bool
            If True, align the indices of `lhs` and `rhs` before performing
            the binary predicate. If False, `lhs` and `rhs` must have the
            same index.

        Returns
        -------
        GeoSeries
            A GeoSeries containing the result of the binary predicate.
        """
        (self.lhs, self.rhs) = lhs.align(rhs) if align else (lhs, rhs)
        self.align = align

    def __call__(self) -> cudf.Series:
        """Return the result of the binary predicate."""
        # Type disambiguation
        # Type disambiguation has a large effect on the decisions of the
        # algorithm.
        (lhs, rhs, indices) = self.preprocess(self.lhs, self.rhs)

        # Binpred call
        point_result = self._op(lhs, rhs)

        # Postprocess: Apply discrete math rules to identify relationships.
        return self.postprocess(indices, point_result)


class ContainsProperlyBinpred(BinaryPredicate):
    def __init__(self, lhs, rhs, align=True, allpairs=False):
        super().__init__(lhs, rhs, align=align)
        self.allpairs = allpairs
        if allpairs:
            self.align = False

    def preprocess(self, lhs, rhs):
        """Preprocess the input GeoSeries to ensure that they are of the
        correct type for the predicate."""
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
        return (lhs, final_rhs, point_indices)

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

    def _op(self, lhs, points):
        """Compute the contains_properly relationship between two GeoSeries.
        A feature A contains another feature B if no points of B lie in the
        exterior of A, and at least one point of the interior of B lies in the
        interior of A. This is the inverse of `within`."""
        if not contains_only_polygons(lhs):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )
        if self._should_use_quadtree():
            point_result = quadtree_contains_properly(
                points,
                lhs,
            )
        else:
            point_result = byte_limited_contains_properly(points, lhs)
        return point_result

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
        ].replace(
            cudf.Series(self.lhs.index, index=cp.arange(len(self.lhs.index)))
        )

        # If the user wants all pairs, return the result. Otherwise,
        # return a boolean series indicating whether each point is
        # contained in the corresponding polygon.
        if self.allpairs:
            return allpairs_result
        else:
            # for each input pair i: result[i] =  true iff point[i] is
            # contained in at least one polygon of multipolygon[i].
            grouped = allpairs_result.groupby("polygon_index").count() >= len(
                point_indices
            )
            final_result = cudf.Series([False] * len(point_indices))
            final_result.loc[grouped.index] = True
            return final_result

    def _postprocess_brute_force_result(self, point_indices, point_result):
        # If there are 31 or fewer polygons in the input, the result
        # is a dataframe with one row per point and one column per
        # polygon.

        # Result can be:
        # A Dataframe of booleans with n_points rows and up to 31 columns.
        # Discrete math recombination
        if (
            contains_only_linestrings(self.rhs)
            or contains_only_polygons(self.rhs)
            or contains_only_multipoints(self.rhs)
        ):
            # process for completed linestrings, polygons, and multipoints.
            # Not necessary for points.
            point_result["idx"] = point_indices
            group_result = (
                point_result.groupby("idx").sum().sort_index()
                == point_result.groupby("idx").count().sort_index()
            )
        else:
            group_result = point_result

        # If there is only one column, the result is a series with
        # one row per point. If it is a dataframe, the result needs
        # to be converted from each matching row/column value to a
        # series using `cp.diag`.
        boolean_series_output = cudf.Series([False] * len(self.lhs))
        boolean_series_output.name = None
        if len(point_result.columns) > 1:
            boolean_series_output[group_result.index] = cp.diag(
                group_result.values
            )
        else:
            boolean_series_output[group_result.index] = group_result[
                group_result.columns[0]
            ]
        return boolean_series_output

    def postprocess(self, point_indices, point_result):
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
        if self._should_use_quadtree():
            return self._postprocess_quadtree_result(
                point_indices, point_result
            )
        else:
            return self._postprocess_brute_force_result(
                point_indices, point_result
            )


class OverlapsBinpred(ContainsProperlyBinpred):
    def postprocess(self, point_indices, point_result):
        # Same as contains_properly, but we need to check that the
        # dimensions are the same.
        # TODO: Maybe change this to intersection
        if not has_same_geometry(self.lhs, self.rhs):
            return cudf.Series([False] * len(self.lhs))
        point_result["point_index"] = point_indices
        hits = point_result.groupby("point_index").sum()
        size = point_result.groupby("point_index").count()
        partial_overlap = hits != size
        non_empty = size > 0
        at_least_one_overlap = hits > 0
        group_result = partial_overlap & non_empty & at_least_one_overlap
        return group_result


class IntersectsBinpred(ContainsProperlyBinpred):
    def preprocess(self, lhs, rhs):
        if contains_only_polygons(rhs):
            (lhs, rhs) = (rhs, lhs)
        return super().preprocess(lhs, rhs)


class WithinBinpred(ContainsProperlyBinpred):
    def preprocess(self, lhs, rhs):
        if contains_only_polygons(rhs):
            (lhs, rhs) = (rhs, lhs)
        return super().preprocess(lhs, rhs)
