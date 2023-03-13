# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from abc import ABC, abstractmethod

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.contains import (
    contains_properly_pairwise,
    contains_properly_quadtree,
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
    def __init__(self, lhs, rhs, align=True, mode="pairs"):
        super().__init__(lhs, rhs, align=align)
        if mode == "allpairs":
            self.allpairs = True
            self.align = False
        else:
            self.allpairs = False

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
        if self.allpairs or has_multipolygons(self.lhs):
            point_result = contains_properly_quadtree(
                points,
                lhs,
            )
        else:
            point_result = contains_properly_pairwise(
                points.points.x,
                points.points.y,
                lhs.polygons.part_offset,
                lhs.polygons.ring_offset,
                lhs.polygons.x,
                lhs.polygons.y,
            )
        return point_result

    def postprocess(self, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the predicate."""
        if self.allpairs or has_multipolygons(self.lhs):

            # This complex block of code is to create a dataframe that
            # contains the polygon index and the point index for each
            # point in the polygon. Quadtree pip returns the _part_index_
            # of the polygon, not the polygon index.
            rings_to_parts = cp.array(self.lhs.polygons.part_offset)
            part_sizes = rings_to_parts[1:] - rings_to_parts[:-1]
            parts_map = cudf.Series(
                cp.arange(len(part_sizes)), name="part_index"
            ).repeat(part_sizes)
            parts_df = parts_map.reset_index(drop=True).reset_index()
            # Mapping of parts to polygons
            parts_to_geoms = cp.array(self.lhs.polygons.geometry_offset)
            geometry_sizes = parts_to_geoms[1:] - parts_to_geoms[:-1]
            geometry_map = cudf.Series(
                cp.arange(len(geometry_sizes)), name="polygon_index"
            ).repeat(geometry_sizes)
            geom_df = geometry_map.reset_index(drop=True)
            geom_df.index.name = "part_index"
            geom_df = geom_df.reset_index()

            # Replace the part index with the polygon index
            part_result = parts_df.merge(point_result, on="part_index")
            # Replace the polygon index with the row index
            result = geom_df.merge(part_result, on="part_index")
            result = result[["polygon_index", "point_index"]]
            result = result.drop_duplicates()
            # Replace the polygon index with the original index
            result["polygon_index"] = result["polygon_index"].replace(
                cudf.Series(
                    self.lhs.index, index=cp.arange(len(self.lhs.index))
                )
            )
            if has_multipolygons(self.lhs) and not self.allpairs:
                if len(result) == 0:
                    return cudf.Series([False] * len(self.lhs))
                final_result = cudf.Series([False] * len(point_indices))
                grouped = result.groupby("polygon_index").count() == len(
                    point_indices
                )
                final_result.loc[grouped.index] = True
                return final_result
            else:
                return result
        else:

            # Result can be:
            # 1. A boolean series
            # 2. A boolean dataframe represeting the relationship between
            #    each point and each polygon.
            #
            # a. Produced by pairwise-point-in-polygon
            # b. Produced by byte-limited-point-in-polygon
            # c. Produced by column-limited-point-in-polygon
            #
            # Aligned or unaligned

            result = point_result
            if has_multipolygons(self.lhs):
                # Recombine the results of the pairwise pip
                # based on the multipolygon index.
                part_indices = self.lhs.polygons.part_offset.take(
                    self.lhs.polygons.geometry_offset
                )
                part_offset = part_indices[1:] - part_indices[:-1]
                part_map = (
                    cudf.Series(cp.arange(len(part_offset)), name="part_index")
                    .repeat(part_offset)
                    .reset_index(drop=True)
                )
                result = point_result
                result["idx"] = part_map
                result = (
                    result.groupby("idx").sum().sort_index()
                    == result.groupby("idx").count().sort_index()
                )

            # Discrete math recombination
            if (
                contains_only_linestrings(self.rhs)
                or contains_only_polygons(self.rhs)
                or contains_only_multipoints(self.rhs)
            ):
                # process for completed linestrings, polygons, and multipoints.
                # Not necessary for points.
                result["idx"] = point_indices
                df_result = (
                    result.groupby("idx").sum().sort_index()
                    == result.groupby("idx").count().sort_index()
                )
                result = df_result
            final_result = result[0]
            final_result.name = None
            return final_result


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
        x = hits != size
        y = size > 0
        z = hits > 0
        group_result = x & y & z
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
