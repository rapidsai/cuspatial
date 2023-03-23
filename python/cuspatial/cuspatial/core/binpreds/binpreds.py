# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from abc import ABC, abstractmethod

import cupy as cp

import cudf

import cuspatial
from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_points,
    contains_only_polygons,
    has_multipolygons,
    has_same_geometry,
)


class PreprocessorOutput:
    """The output of the preprocess method of a binary predicate.

    This makes it possible to create a class that matches the necessary
    signature of a geoseries.GeoColumnAccessor object. In some cases the
    preprocessor may need to reorder the input data, in which case the
    preprocessor will return a PreprocessorOutput object instead of a
    GeoColumnAccessor."""

    def __init__(self, coords, indices) -> None:
        self.vertices = coords
        self.indices = indices

    @property
    def xy(self):
        return self.vertices

    def point_indices(self):
        return self.indices


class BinaryPredicate(ABC):
    """BinaryPredicate is an abstract class that implements the binary
    predicate algorithm. The binary predicate algorithm is used to compute
    the relationship between two GeoSeries.

    The algorithm is implemented in three steps: `preprocess`, `_op`, and
    `postprocess`. The `preprocess` step is used to ensure that the input
    GeoSeries are of the correct type for the binary predicate. The
    `_op` step is used to compute the relationship between the points
    in the input GeoSeries. The `postprocess` step is used to compute
    the relationship between the input GeoSeries.
    """

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
        return (lhs, rhs, cudf.RangeIndex(len(rhs)))

    @abstractmethod
    def postprocess(
        self, point_indices: cudf.Series, point_result: cudf.Series
    ) -> cudf.Series:
        """Postprocess the output data for the binary predicate. This method
        should be implemented by subclasses.

        Postprocess converts the raw results of the binary predicate into
        the final result. This is where the discrete math rules are applied.
        The binary predicate operation does not compute any relationships
        between features in the input GeoSeries', it only computes the
        relationship between the points in the input geometries. The
        postprocess method uses the discrete math rules to compute the
        relationship between the input geometries.

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

    def _cancel_op(self, lhs, rhs, result):
        """Used to disable computation of the binary predicate.

        This occurs when the binary predicate is not supported for the
        input types, and a final result can be computed only using
        `preprocess` and `postprocess`."""
        self._op = lambda x, y: result
        return (lhs, rhs, result)

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
        final_rhs = cuspatial.core.geoseries.GeoSeries(
            GeoColumn._from_points_xy(xy_points._column)
        )
        return (lhs, final_rhs, cudf.Series(point_indices))

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
            return contains_properly(lhs, points, how="quadtree")
        else:
            return contains_properly(lhs, points, how="byte-limited")

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
        if not contains_only_polygons(self.lhs):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )
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

    def _count_results_in_multipoint_geometries(
        self, point_indices, point_result
    ):
        """Count the number of points in each multipoint geometry.

        Parameters
        ----------
        point_indices : cudf.Series
            The indices of the points in the original (rhs) GeoSeries.
        point_result : cudf.DataFrame
            The result of a contains_properly call.

        Returns
        -------
        cudf.Series
            The number of points that fell within a particular polygon id.
        cudf.Series
            The number of points in each multipoint geometry.
        """
        point_indices_df = cudf.Series(
            point_indices,
            name="rhs_index",
            index=cudf.RangeIndex(len(point_indices), name="point_index"),
        ).reset_index()
        with_rhs_indices = point_result.merge(
            point_indices_df, on="point_index"
        )
        points_grouped_by_original_polygon = with_rhs_indices[
            ["point_index", "rhs_index"]
        ].drop_duplicates()
        hits = (
            points_grouped_by_original_polygon.groupby("rhs_index")
            .count()
            .sort_index()
        )
        expected_count = (
            point_indices_df.groupby("rhs_index").count().sort_index()
        )
        return hits, expected_count

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
                    final_result = cudf.Series([False] * len(point_indices))
                    final_result.loc[
                        allpairs_result["polygon_index"][matches]
                    ] = True
                    return final_result
                else:
                    final_result = cudf.Series([False] * len(point_indices))
                    final_result.loc[allpairs_result["polygon_index"]] = True
                    return final_result

    def _postprocess_brute_force_result(self, point_indices, point_result):
        # If there are 31 or fewer polygons in the input, the result
        # is a dataframe with one row per point and one column per
        # polygon.

        # Result can be:
        # A Dataframe of booleans with n_points rows and up to 31 columns.
        if (
            contains_only_linestrings(self.rhs)
            or contains_only_polygons(self.rhs)
            or contains_only_multipoints(self.rhs)
        ):
            # Compute the set of results for each point-in-polygon predicate.
            # Group them by the original index, and sum the results. If the
            # sum of points in the rhs feature is equal to the number of
            # points found in the polygon, then the polygon contains the
            # feature.
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
        return self._postprocess_quadtree_result(point_indices, point_result)


class OverlapsBinpred(ContainsProperlyBinpred):
    def postprocess(self, point_indices, point_result):
        # Same as contains_properly, but we need to check that the
        # dimensions are the same.
        # TODO: Maybe change this to intersection
        if not has_same_geometry(self.lhs, self.rhs):
            return cudf.Series([False] * len(self.lhs))
        if len(point_result) == 0:
            return cudf.Series([False] * len(self.lhs))
        polygon_indices = (
            self._convert_quadtree_result_from_part_to_polygon_indices(
                point_result
            )
        )
        group_counts = polygon_indices.groupby("polygon_index").count()
        point_counts = (
            cudf.DataFrame(
                {"point_indices": point_indices, "input_size": True}
            )
            .groupby("point_indices")
            .count()
        )
        result = (group_counts["point_index"] > 0) & (
            group_counts["point_index"] < point_counts["input_size"]
        )
        return result


class IntersectsBinpred(ContainsProperlyBinpred):
    def preprocess(self, lhs, rhs):
        if contains_only_polygons(rhs):
            (self.lhs, self.rhs) = (rhs, lhs)
        return super().preprocess(rhs, lhs)

    def postprocess(self, point_indices, point_result):
        # Same as contains_properly, but we need to check that the
        # dimensions are the same.
        contains_result = super().postprocess(point_indices, point_result)
        return contains_result


class WithinBinpred(ContainsProperlyBinpred):
    def preprocess(self, lhs, rhs):
        if contains_only_polygons(rhs):
            (self.lhs, self.rhs) = (rhs, lhs)
        return super().preprocess(rhs, lhs)

    def postprocess(self, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        (hits, expected_count,) = self._count_results_in_multipoint_geometries(
            point_indices, point_result
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


class EqualsBinpred(BinaryPredicate):
    def _offset_equals(self, lhs, rhs):
        """Compute the pairwise length equality of two offset arrays"""
        lhs_lengths = lhs[:-1] - lhs[1:]
        rhs_lengths = rhs[:-1] - rhs[1:]
        return lhs_lengths == rhs_lengths

    def _sort_interleaved_points_by_offset(self, coords, offsets, sort_order):
        """Sort xy according to bins defined by offset. Sort order is a list
        of column names to sort by.

        `_sort_interleaved_points_by_offset` creates a dataframe with the
        following columns:
        "sizes": an index for each object represented in `coords`.
        "points": an index for each point in `coords`.
        "xy_key": an index that maintains x/y ordering.
        "xy": the x/y coordinates in `coords`.

        The dataframe is sorted according to keys passed in by the caller.
        For sorting multipoints, the keys in order are "object_key", "xy",
        "xy_key". This sorts the points in each multipoint into the same
        bin defined by "object_key", then sorts the points in each bin by
        x/y coordinates, and finally sorts the points in each bin by the
        `xy_key` which maintains that the x coordinate precedes the y
        coordinate.

        For sorting linestrings, the keys in order are "object_key",
        "point_key", "xy_key". This sorts the points in each linestring
        into the same bin defined by "object_key", then sorts the points
        in each bin by point ordering, and finally sorts the points in
        each bin by x/y ordering.

        Parameters
        ----------
        coords : cudf.Series
            interleaved x,y coordinates
        offsets : cudf.Series
            offsets into coords
        sort_order : list
            list of column names to sort by. One of "object_key", "point_key",
            "xy_key", and "xy".

        Returns
        -------
        cudf.Series
            sorted interleaved x,y coordinates
        """
        sizes = offsets[1:] - offsets[:-1]
        object_key = (
            cudf.Series(cp.arange(len(sizes)))
            .repeat(sizes * 2)
            .reset_index(drop=True)
        )
        point_key = cp.arange(len(coords) // 2).repeat(2)[::-1]
        xy_key = cp.tile([0, 1], len(coords) // 2)
        sorting_df = cudf.DataFrame(
            {
                "object_key": object_key,
                "point_key": point_key,
                "xy_key": xy_key,
                "xy": coords,
            }
        )
        sorted_df = sorting_df.sort_values(by=sort_order).reset_index(
            drop=True
        )
        return sorted_df["xy"]

    def _sort_multipoint_series(self, coords, offsets):
        """Sort xy according to bins defined by offset"""
        result = self._sort_interleaved_points_by_offset(
            coords, offsets, ["object_key", "xy", "xy_key"]
        )
        result.name = None
        return result

    def _sort_multipoints(self, lhs, rhs, initial):
        lhs_sorted = self._sort_multipoint_series(
            lhs.multipoints.xy, lhs.multipoints.geometry_offset
        )
        rhs_sorted = self._sort_multipoint_series(
            rhs.multipoints.xy, rhs.multipoints.geometry_offset
        )
        lhs_result = cuspatial.core.geoseries.GeoSeries.from_multipoints_xy(
            lhs_sorted, lhs.multipoints.geometry_offset
        )
        rhs_result = cuspatial.core.geoseries.GeoSeries.from_multipoints_xy(
            rhs_sorted, rhs.multipoints.geometry_offset
        )
        lhs_result.index = lhs.index
        rhs_result.index = rhs.index
        return (
            lhs_result,
            rhs_result,
            initial,
        )

    def _reverse_linestrings(self, coords, offsets):
        """Reverse the order of coordinates in a Arrow buffer of coordinates
        and offsets."""
        result = self._sort_interleaved_points_by_offset(
            coords, offsets, ["object_key", "point_key", "xy_key"]
        )
        result.name = None
        return result

    def _compare_linestrings_and_reversed(self, lhs, rhs, initial):
        """Compare linestrings with their reversed counterparts."""
        lhs_xy = self._reverse_linestrings(lhs.xy, lhs.part_offset)
        rhs_xy = self._reverse_linestrings(rhs.xy, rhs.part_offset)
        return (
            PreprocessorOutput(lhs_xy, lhs.point_indices()),
            PreprocessorOutput(rhs_xy, rhs.point_indices()),
            initial,
        )

    def preprocess(self, lhs, rhs):
        # Compare types
        type_compare = lhs.feature_types == rhs.feature_types
        # Any unmatched type is not equal
        if (type_compare == False).all():  # noqa: E712
            # Override _op so that it will not be run.
            return self._cancel_op(lhs, rhs, type_compare)
        # Get indices of matching types
        if contains_only_multipoints(lhs):
            lengths_equal = self._offset_equals(
                lhs.multipoints.geometry_offset,
                rhs.multipoints.geometry_offset,
            )
            if lengths_equal.any():
                # Multipoints are equal if they contains the
                # same unordered points.
                return self._sort_multipoints(
                    lhs[lengths_equal],
                    rhs[lengths_equal],
                    lengths_equal,
                )
            else:
                # No lengths are equal, so none can be equal.
                return self._cancel_op(lhs, rhs, lengths_equal)
        elif contains_only_linestrings(lhs):
            lengths_equal = self._offset_equals(
                lhs.lines.part_offset, rhs.lines.part_offset
            )
            if lengths_equal.any():
                return (
                    lhs[lengths_equal],
                    rhs[lengths_equal],
                    lengths_equal,
                )
            else:
                return self._cancel_op(lhs, rhs, lengths_equal)
        elif contains_only_polygons(lhs):
            raise NotImplementedError
        elif contains_only_points(lhs):
            return (lhs, rhs, type_compare)

    def postprocess(self, lengths_equal, point_result):
        # if point_result is not a Series, preprocessing terminated
        # the results early.
        if isinstance(point_result, cudf.Series):
            point_result = point_result.sort_index()
            lengths_equal[point_result.index] = point_result
        return cudf.Series(lengths_equal)

    def _vertices_equals(self, lhs, rhs):
        """Compute the equals relationship between interleaved xy
        coordinate buffers."""
        length = min(len(lhs), len(rhs))
        a = lhs[:length:2]._column == rhs[:length:2]._column
        b = rhs[1:length:2]._column == lhs[1:length:2]._column
        return a & b

    def _op(self, lhs, rhs):
        if contains_only_linestrings(lhs):
            # Linestrings can be compared either forward or
            # reversed. We need to compare both directions.
            lhs_reversed = self._reverse_linestrings(
                lhs.lines.xy, lhs.lines.part_offset
            )
            forward_result = self._vertices_equals(lhs.lines.xy, rhs.lines.xy)
            reverse_result = self._vertices_equals(lhs_reversed, rhs.lines.xy)
            result = forward_result | reverse_result
        elif contains_only_multipoints(lhs):
            result = self._vertices_equals(
                lhs.multipoints.xy, rhs.multipoints.xy
            )
        elif contains_only_points(lhs):
            result = self._vertices_equals(lhs.points.xy, rhs.points.xy)
        elif contains_only_polygons(lhs):
            raise NotImplementedError
        indices = lhs.point_indices
        result_df = cudf.DataFrame(
            {"idx": indices[: len(result)], "equals": result}
        )
        gb_idx = result_df.groupby("idx")
        result = (gb_idx.sum().sort_index() == gb_idx.count().sort_index())[
            "equals"
        ]
        result.index = lhs.index
        result.index.name = None
        result.name = None
        return result


class CrossesBinpred(EqualsBinpred):
    """An object is said to cross other if its interior intersects the
    interior of the other but does not contain it, and the dimension of
    the intersection is less than the dimension of the one or the other.

    This is only implemented for `points.crosses(points)` at this time.
    """

    def postprocess(self, point_indices, point_result):
        if has_same_geometry(self.lhs, self.rhs) and contains_only_points(
            self.lhs
        ):
            return cudf.Series([False] * len(self.lhs))
        df_result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result


class CoversBinpred(EqualsBinpred):
    """An object A covers another B if no points of B lie in the exterior
    of A, and at least one point of the interior of B lies in the interior
    of A.

    This is only implemented for `points.covers(points)` at this time."""

    def postprocess(self, point_indices, point_result):
        return cudf.Series(point_result, index=point_indices)
