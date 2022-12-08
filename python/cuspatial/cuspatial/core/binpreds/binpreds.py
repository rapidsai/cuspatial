# Copyright (c) 2022, NVIDIA CORPORATION.

from abc import ABC, abstractmethod

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.contains import contains_properly
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_points,
    contains_only_polygons,
    has_same_dimension,
)


class BinaryPredicate(ABC):
    @abstractmethod
    def preprocess(self, op, lhs, rhs):
        """Preprocess the input data for the binary operation. This method
        should be implemented by subclasses. Preprocess and postprocess are
        used to implement the discrete math rules of the binary operations.

        Notes
        -----
        Should only be called once.

        Parameters
        ----------
        op : str
            The binary operation to perform.
        lhs : GeoSeries
            The left-hand-side of the GeoSeries-level binary operation.
        rhs : GeoSeries
            The right-hand-side of the GeoSeries-level binary operation.

        Returns
        -------
        GeoSeries
            The left-hand-side of the internal binary operation, may be
            reordered.
        GeoSeries
            The right-hand-side of the internal binary operation, may be
            reordered.
        """
        pass

    @abstractmethod
    def postprocess(self, op, point_indices, point_result):
        """Postprocess the output data for the binary operation. This method
        should be implemented by subclasses.

        Postprocess converts the raw results of the binary operation into
        the final result. This is where the discrete math rules are applied.

        Parameters
        ----------
        op : str
            The binary operation to post process. Determines for example the
            set operation to use for computing the result.
        processed : cudf.Series
            The data after applying the fundamental binary operation.

        Returns
        -------
        cudf.Series
            The output of the post processing, True/False results for
            the specified binary op.
        """
        pass

    def __init__(self, op, lhs, rhs, align=True):
        """Compute the binary operation `op` on `lhs` and `rhs`.

        There are ten binary operations supported by cuspatial:
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
        operation is not symmetric. For example, `A.contains(B)` is not
        necessarily the same as `B.within(A)`.

        Parameters
        ----------
        op : str
            The binary operation to perform.
        lhs : GeoSeries
            The left-hand-side of the binary operation.
        rhs : GeoSeries
            The right-hand-side of the binary operation.
        align : bool
            If True, align the indices of `lhs` and `rhs` before performing
            the binary operation. If False, `lhs` and `rhs` must have the
            same index.

        Returns
        -------
        GeoSeries
            A GeoSeries containing the result of the binary operation.
        """
        (self.lhs, self.rhs) = lhs.align(rhs) if align else (lhs, rhs)
        self.align = align

        # Type disambiguation
        # Type disambiguation has a large effect on the decisions of the
        # algorithm.
        (lhs, rhs, indices) = self.preprocess(op, self.lhs, self.rhs)

        # Binpred call
        _binpred = getattr(self, op)
        point_result = _binpred(lhs, rhs)

        # Postprocess: Apply discrete math rules to identify relationships.
        final_result = self.postprocess(op, indices, point_result)

        # Return
        self.op_result = final_result

    def __call__(self) -> cudf.Series:
        """Return the result of the binary operation."""
        return self.op_result

    def contains_properly(self, lhs, points):
        """Compute the contains_properly relationship between two GeoSeries.
        A feature A contains another feature B if no points of B lie in the
        exterior of A, and at least one point of the interior of B lies in the
        interior of A. This is the inverse of `within`."""
        if contains_only_points(lhs) and contains_only_points(points):
            return self.equals(lhs, points)
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

    def equals(self, lhs, rhs):
        """Compute the equals relationship between two GeoSeries."""
        result = False
        if contains_only_points(lhs):
            result = lhs.points.xy.equals(rhs.points.xy)
        elif contains_only_linestrings(lhs):
            result = lhs.lines.xy.equals(rhs.lines.xy)
        elif contains_only_polygons(lhs):
            result = lhs.polygons.xy.equals(rhs.polygons.xy)
        elif contains_only_multipoints(lhs):
            result = lhs.multipoints.xy.equals(rhs.multipoints.xy)
        return result

    def touches(self, lhs, rhs):
        """Compute the touches relationship between two GeoSeries:
        Two objects touch if they share at least one point in common, but their
        interiors do not intersect."""
        return self.equals(lhs, rhs)

    def covers(self, lhs, rhs):
        """A covers B if no points of B are in the exterior of A."""
        return self.contains_properly(lhs, rhs)

    def intersects(self, lhs, rhs):
        """Compute from a GeoSeries of points and a GeoSeries of polygons which
        points are contained within the corresponding polygon. Polygon A
        contains Point B if B intersects the interior or boundary of A.
        """
        # TODO: If rhs is polygon and lhs is point, use contains_properly
        return self.contains_properly(lhs, rhs)

    def within(self, lhs, rhs):
        """An object is said to be within rhs if at least one of its points
        is located in the interior and no points are located in the exterior
        of the rhs."""
        return self.contains_properly(lhs, rhs)

    def crosses(self, lhs, rhs):
        """Returns a `Series` of `dtype('bool')` with value `True` for each
        aligned geometry that crosses rhs.

        An object is said to cross rhs if its interior intersects the
        interior of the rhs but does not contain it, and the dimension of
        the intersection is less than either."""
        # Crosses requires the use of point_in_polygon but only requires that
        # 1 or more points are within the polygon. This differs from
        # `.contains` which requires all of them.
        return self.contains_properly(lhs, rhs)

    def overlaps(self, lhs, rhs, align=True):
        """Returns True for all aligned geometries that overlap rhs, else
        False.

        Geometries overlap if they have more than one but not all points in
        common, have the same dimension, and the intersection of the
        interiors of the geometries has the same dimension as the geometries
        themselves."""
        # Overlaps has the same requirement as crosses.
        return self.contains_properly(lhs, rhs)


class ContainsProperlyBinpred(BinaryPredicate):
    def preprocess(self, op, lhs, rhs):
        """Preprocess the input GeoSeries to ensure that they are of the
        correct type for the operation."""
        # Type determines discrete math recombination outcome
        if contains_only_points(lhs) and contains_only_points(rhs):
            return (lhs, rhs, rhs.points.point_indices())
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

        # Arrange into shape for calling pip, intersection, or equals
        point_indices = geom.point_indices()
        from cuspatial.core.geoseries import GeoSeries

        final_rhs = GeoSeries(
            GeoColumn._from_points_xy(xy_points._column)
        ).points
        return (lhs, final_rhs, point_indices)

    def postprocess(self, op, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the operation."""
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


class OverlapsBinpred(ContainsProperlyBinpred):
    def postprocess(self, op, point_indices, point_result):
        # Same as contains_properly, but we need to check that the
        # dimensions are the same.
        # TODO: Maybe change this to intersection
        if not has_same_dimension(self.lhs, self.rhs):
            return cudf.Series([False] * len(self.lhs))
        result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        df_result = result
        # Discrete math recombination
        if contains_only_linestrings(self.rhs):
            df_result = (
                result.groupby("idx").sum().sort_index()
                == result.groupby("idx").count().sort_index()
            )
        elif contains_only_polygons(self.rhs) or contains_only_multipoints(
            self.rhs
        ):
            df_result = result.groupby("idx").sum().sort_index() > 0
        else:
            df_result = result.groupby("idx").sum().sort_index() > 1
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result


class IntersectsBinpred(ContainsProperlyBinpred):
    def preprocess(self, op, lhs, rhs):
        if contains_only_polygons(rhs):
            (lhs, rhs) = (rhs, lhs)
        return super().preprocess(op, lhs, rhs)

    def postprocess(self, op, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the operation."""
        return super().postprocess(op, point_indices, point_result)


class WithinBinpred(ContainsProperlyBinpred):
    def preprocess(self, op, lhs, rhs):
        if contains_only_polygons(rhs):
            (lhs, rhs) = (rhs, lhs)
        return super().preprocess(op, lhs, rhs)

    def postprocess(self, op, point_indices, point_result):
        """Postprocess the output GeoSeries to ensure that they are of the
        correct type for the operation."""
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
        if op == "equals":
            if len(point_result) == 1:
                return point_result[0]
        return point_result


class CrossesBinpred(BinaryPredicate):
    def postprocess(self, op, point_indices, point_result):
        result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        df_result = result
        # Discrete math recombination
        if (
            contains_only_linestrings(self.rhs)
            or contains_only_polygons(self.rhs)
            or contains_only_multipoints(self.rhs)
        ):
            df_result = ~result
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result


class EqualsBinpred(ContainsProperlyBinpred):
    def postprocess(self, op, point_indices, point_result):
        if len(point_result) == 1:
            return point_result[0]
        return point_result


class CoversBinpred(ContainsProperlyBinpred):
    pass


class TouchesBinpred(ContainsProperlyBinpred):
    pass
