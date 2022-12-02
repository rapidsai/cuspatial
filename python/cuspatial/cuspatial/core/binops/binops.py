# Copyright (c) 2022, NVIDIA CORPORATION.

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binops.contains import contains_properly
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
    has_same_dimension,
)


class _binop:
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

        There are twenty-five ordering combinations of `lhs` and `rhs`:
        - point     point
        - point     multipoint
        - point     linestring
        - point     multilinestring
        - point     polygon
        - point     multipolygon
        - multipoint point
        - multipoint multipoint
        - multipoint linestring
        - multipoint multilinestring
        - multipoint polygon
        - multipoint multipolygon
        - linestring point
        - linestring multipoint
        - linestring linestring
        - linestring multilinestring
        - linestring polygon
        - linestring multipolygon
        - multilinestring point
        - multilinestring multipoint
        - multilinestring linestring
        - multilinestring multilinestring
        - multilinestring polygon
        - multilinestring multipolygon
        - polygon point
        - polygon multipoint
        - polygon linestring
        - polygon multilinestring
        - polygon polygon
        - polygon multipolygon
        - multipolygon point
        - multipolygon multipoint
        - multipolygon linestring
        - multipolygon multilinestring
        - multipolygon polygon
        - multipolygon multipolygon

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
        if self.is_invalid_combination(op, lhs, rhs):
            self.op_result = cudf.Series([False] * len(lhs))
            return

        (self.lhs, self.rhs) = lhs.align(rhs) if align else (lhs, rhs)
        self.align = align

        # Type disambiguation
        # Type disambiguation has a large effect on the decisions of the
        # algorithm.
        (self.lhs, self.rhs) = self.preprocess(op, self.lhs, self.rhs)
        # Type determines discrete math recombination outcome
        # RHS conditioning:
        mode = "POINTS"
        # point in polygon
        if contains_only_linestrings(self.rhs):
            # condition for linestrings
            mode = "LINESTRINGS"
            geom = self.rhs.lines
        elif contains_only_polygons(self.rhs) is True:
            # polygon in polygon
            mode = "POLYGONS"
            geom = self.rhs.polygons
        elif contains_only_multipoints(self.rhs) is True:
            # mpoint in polygon
            mode = "MULTIPOINTS"
            geom = self.rhs.multipoints
        else:
            # no conditioning is required
            geom = self.rhs.points
        xy_points = geom.xy

        # Arrange into shape for calling pip, intersection, or equals
        point_indices = geom.point_indices()
        from cuspatial.core.geoseries import GeoSeries

        points = GeoSeries(GeoColumn._from_points_xy(xy_points._column)).points

        # Binop call
        _binop = getattr(self, op)
        point_result = _binop(self.lhs, points)

        # Postprocess: Apply discrete math rules to identify relationships.
        final_result = self.postprocess(op, point_indices, point_result, mode)

        # Return
        self.op_result = final_result

    def __call__(self) -> cudf.Series:
        return self.op_result

    def is_invalid_combination(self, op, lhs, rhs):
        """Many operations return false if the input is not valid.
        Binary operations that depend on a particular dimensionality are the
        first case implemented here."""
        if op == "overlaps":
            if not has_same_dimension(lhs, rhs):
                return True
        return False

    def preprocess(self, op, lhs, rhs):
        """Preprocess the input data for the binary operation.

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
            The left-hand-side of the internal binary operation.
        GeoSeries
            The right-hand-side of the internal binary operation.
        """
        if op == "intersects" or op == "within":
            if contains_only_polygons(rhs):
                return (rhs, lhs)
        return (lhs, rhs)

    def postprocess(self, op, point_indices, point_result, mode):
        """Postprocess the output data for the binary operation.

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
        result = cudf.DataFrame({"idx": point_indices, "pip": point_result})
        df_result = result
        # Discrete math recombination
        if (
            mode == "LINESTRINGS"
            or mode == "POLYGONS"
            or mode == "MULTIPOINTS"
        ):
            # process for completed linestrings, polygons, and multipoints.
            # Not necessary for points.
            if op == "contains" or op == "contains_properly" or op == "within":
                df_result = (
                    result.groupby("idx").sum().sort_index()
                    == result.groupby("idx").count().sort_index()
                )
            if op == "overlaps":
                if mode == "LINESTRINGS":
                    df_result = (
                        result.groupby("idx").sum().sort_index()
                        == result.groupby("idx").count().sort_index()
                    )
                else:
                    df_result = result.groupby("idx").sum().sort_index() > 0
        else:
            # Points only
            if op == "overlaps":
                df_result = result.groupby("idx").sum().sort_index() > 1
            elif op == "crosses":
                df_result = ~result
        point_result = cudf.Series(
            df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
        )
        point_result.name = None
        return point_result

    def contains_properly(self, lhs, points):
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

    def covers(self, lhs, rhs, align=True):
        """A covers B if no points of B are in the exterior of A."""
        return self.contains_properly(lhs, rhs, align=align)

    def intersects(self, lhs, rhs, align=True):
        """Compute from a GeoSeries of points and a GeoSeries of polygons which
        points are contained within the corresponding polygon. Polygon A
        contains Point B if B intersects the interior or boundary of A.

        Parameters
        ----------
        rhs
            a cuspatial.GeoSeries
        """
        # TODO: If rhs is polygon and lhs is point, use contains_properly
        return self.contains_properly(lhs, rhs)

    def within(self, lhs, rhs, align=True):
        """An object is said to be within rhs if at least one of its points
        is located in the interior and no points are located in the exterior
        of the rhs."""
        return self.contains_properly(lhs, rhs)

    def crosses(self, lhs, rhs, align=True):
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
