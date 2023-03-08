# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from abc import ABC, abstractmethod

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_points,
    contains_only_polygons,
    has_same_geometry,
)


class PreprocessorOutput:
    def __init__(self, coords, indices) -> None:
        self.vertices = coords
        self.indices = indices

    @property
    def xy(self):
        return self.vertices

    def point_indices(self):
        return self.indices


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
        return (lhs, rhs, cudf.RangeIndex(len(rhs)))

    @abstractmethod
    def postprocess(
        self, point_indices: cudf.Series, point_result: cudf.Series
    ) -> cudf.Series:
        """Postprocess the output data for the binary predicate. This method
        should be implemented by subclasses.

        Postprocess converts the raw results of the binary predicate into
        the final result. At this step the results for none, any, and all
        are applied to the result of the equals, intersects, and
        point-in-polygon predicates.

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

        final_rhs = GeoSeries(
            GeoColumn._from_points_xy(xy_points._column)
        ).points
        return (lhs, final_rhs, point_indices)

    def _op(self, lhs, points):
        """Compute the contains_properly relationship between two GeoSeries.
        A feature A contains another feature B if no points of B lie in the
        exterior of A, and at least one point of the interior of B lies in the
        interior of A. This is the inverse of `within`."""
        if not contains_only_polygons(lhs):
            raise TypeError(
                "`.contains` can only be called with polygon series."
            )

        # call pip on the three subtypes on the right:
        point_result = contains_properly(
            points.x,
            points.y,
            lhs.polygons.part_offset[:-1],
            lhs.polygons.ring_offset[:-1],
            lhs.polygons.x,
            lhs.polygons.y,
        )
        return point_result

    def postprocess(self, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        df_result = result
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
            df_result = (
                result.groupby("idx").sum().sort_index()
                == result.groupby("idx").count().sort_index()
            )
        # Convert the result to a GeoSeries.
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result


class OverlapsBinpred(ContainsProperlyBinpred):
    def postprocess(self, point_indices, point_result):
        # Same as contains_properly, but we need to check that the
        # dimensions are the same.
        # TODO: Maybe change this to intersection
        if not has_same_geometry(self.lhs, self.rhs):
            return cudf.Series([False] * len(self.lhs))
        result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        df_result = result
        partial_result = result.groupby("idx").sum()
        df_result = (partial_result > 0) & (partial_result < len(point_result))
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result


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

    def postprocess(self, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        df_result = result
        # Discrete math recombination
        if (
            contains_only_linestrings(self.rhs)
            or contains_only_polygons(self.rhs)
            or contains_only_multipoints(self.rhs)
        ):
            # process for completed linestrings, polygons, and multipoints.
            # Not necessary for points.
            df_result = (
                result.groupby("idx").sum().sort_index()
                == result.groupby("idx").count().sort_index()
            )
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result


class EqualsBinpred(BinaryPredicate):
    def _offset_equals(self, lhs, rhs):
        """Compute the pairwise length equality of two offset arrays"""
        lhs_lengths = lhs[:-1] - lhs[1:]
        rhs_lengths = rhs[:-1] - rhs[1:]
        return lhs_lengths == rhs_lengths

    def _sort_multipoints(self, lhs, rhs, initial):
        """Sort xy according to bins defined by offset"""
        sort_indices = cp.repeat(lhs.point_indices(), 2)
        lhs_xy, rhs_xy = lhs.xy, rhs.xy
        lhs_xy.index = sort_indices
        rhs_xy.index = sort_indices
        lhs_df = lhs_xy.reset_index(drop=False, name="xy")
        rhs_df = rhs_xy.reset_index(drop=False, name="xy")
        lhs_sorted = lhs_df.sort_values(by=["index", "xy"]).reset_index(
            drop=True
        )
        rhs_sorted = rhs_df.sort_values(by=["index", "xy"]).reset_index(
            drop=True
        )
        return (
            PreprocessorOutput(lhs_sorted["xy"], lhs.point_indices()),
            PreprocessorOutput(rhs_sorted["xy"], rhs.point_indices()),
            initial,
        )

    def _sort_linestrings(self, lhs, rhs, initial):
        """Swap first and last values of each linestring to ensure that
        the first point is the lowest value. This is necessary to ensure
        that the endpoints are not included in the comparison."""
        # Save temporary values since lhs.x cannot be used modified.
        lhs_x = lhs.x
        lhs_y = lhs.y
        rhs_x = rhs.x
        rhs_y = rhs.y
        point_range = cudf.Series(cp.arange(len(lhs_x)))
        indices = point_range.groupby(lhs.point_indices())
        # Create masks for the first and last values of each linestring
        swap_lx = lhs_x[indices.last()].reset_index(drop=True) < lhs_x[
            indices.first()
        ].reset_index(drop=True)
        swap_ly = lhs_y[indices.last()].reset_index(drop=True) < lhs_y[
            indices.first()
        ].reset_index(drop=True)
        swap_rx = rhs_x[indices.last()].reset_index(drop=True) < rhs_x[
            indices.first()
        ].reset_index(drop=True)
        swap_ry = rhs_y[indices.last()].reset_index(drop=True) < rhs_y[
            indices.first()
        ].reset_index(drop=True)
        # Swap the first and last values of each linestring
        (
            lhs_x.iloc[indices.last()[swap_lx]],
            lhs_x.iloc[indices.first()[swap_lx]],
        ) = (
            lhs_x.iloc[indices.first()[swap_lx]],
            lhs_x.iloc[indices.last()[swap_lx]],
        )
        (
            lhs_y.iloc[indices.last()[swap_ly]],
            lhs_y.iloc[indices.first()[swap_ly]],
        ) = (
            lhs_y.iloc[indices.first()[swap_ly]],
            lhs_y.iloc[indices.last()[swap_ly]],
        )
        (
            rhs_x.iloc[indices.last()[swap_rx]],
            rhs_x.iloc[indices.first()[swap_rx]],
        ) = (
            rhs_x.iloc[indices.first()[swap_rx]],
            rhs_x.iloc[indices.last()[swap_rx]],
        )
        (
            rhs_y.iloc[indices.last()[swap_ry]],
            rhs_y.iloc[indices.first()[swap_ry]],
        ) = (
            rhs_y.iloc[indices.first()[swap_ry]],
            rhs_y.iloc[indices.last()[swap_ry]],
        )
        # Reconstruct the xy columns
        lhs_xy = lhs.xy
        rhs_xy = rhs.xy
        lhs_xy.iloc[::2] = lhs_x
        lhs_xy.iloc[1::2] = lhs_y
        rhs_xy.iloc[::2] = rhs_x
        rhs_xy.iloc[1::2] = rhs_y

        return (
            PreprocessorOutput(lhs_xy, lhs.point_indices()),
            PreprocessorOutput(rhs_xy, rhs.point_indices()),
            initial,
        )

    def preprocess(self, lhs, rhs):
        # Compare types
        type_compare = lhs.dtype == rhs.dtype
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
                    lhs[lengths_equal].multipoints,
                    rhs[lengths_equal].multipoints,
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
                # Linestrings are equal if their sorted points
                # are equal. This is unintuitive and perhaps
                # incorrect, but it is the behavior of shapely.
                return self._sort_linestrings(
                    lhs[lengths_equal].lines,
                    rhs[lengths_equal].lines,
                    lengths_equal,
                )
            else:
                return self._cancel_op(lhs, rhs, lengths_equal)
        elif contains_only_polygons(lhs):
            geoms_equal = self._offset_equals(
                lhs.polygons.part_offset, rhs.polygons.part_offset
            )
            lengths_equal = self._offset_equals(
                lhs[geoms_equal].polygons.ring_offset,
                rhs[geoms_equal].polygons.ring_offset,
            )
            if lengths_equal.any():
                # Don't sort polygons
                return (
                    lhs[lengths_equal].polygons,
                    rhs[lengths_equal].polygons,
                    lengths_equal,
                )
            else:
                return self._cancel_op(lhs, rhs, lengths_equal)
        elif contains_only_points(lhs):
            return (lhs.points, rhs.points, type_compare)

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
        indices = lhs.point_indices()
        result = self._vertices_equals(lhs.xy, rhs.xy)
        result_df = cudf.DataFrame(
            {"idx": indices[: len(result)], "equals": result}
        )
        gb_idx = result_df.groupby("idx")
        result = (gb_idx.sum().sort_index() == gb_idx.count().sort_index())[
            "equals"
        ]
        result.index.name = None
        result.name = None
        return result


class CrossesBinpred(EqualsBinpred):
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
    def postprocess(self, point_indices, point_result):
        return cudf.Series(point_result, index=point_indices)
