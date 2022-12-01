# Copyright (c) 2022, NVIDIA CORPORATION.

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binops.contains import contains_properly
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
)


class _binop:
    def __init__(self, op, lhs, rhs, align=True):
        (self.lhs, self.rhs) = lhs.align(rhs) if align else (lhs, rhs)
        self.align = align

        # Type disambiguation
        # Type determines discrete math recombination outcome
        # RHS conditioning:
        mode = "POINTS"
        # point in polygon
        if contains_only_linestrings(rhs):
            # condition for linestrings
            mode = "LINESTRINGS"
            geom = rhs.lines
        elif contains_only_polygons(rhs) is True:
            # polygon in polygon
            mode = "POLYGONS"
            geom = rhs.polygons
        elif contains_only_multipoints(rhs) is True:
            # mpoint in polygon
            mode = "MULTIPOINTS"
            geom = rhs.multipoints
        else:
            # no conditioning is required
            geom = rhs.points
        xy_points = geom.xy

        # Arrange into shape for calling pip, intersection, or equals
        point_indices = geom.point_indices()
        from cuspatial.core.geoseries import GeoSeries

        points = GeoSeries(GeoColumn._from_points_xy(xy_points._column)).points

        # Binop call
        _binop = getattr(self, op)
        point_result = _binop(lhs, points)

        # Discrete math recombination
        if (
            mode == "LINESTRINGS"
            or mode == "POLYGONS"
            or mode == "MULTIPOINTS"
        ):
            # process for completed linestrings, polygons, and multipoints.
            # Not necessary for points.
            result = cudf.DataFrame(
                {"idx": point_indices, "pip": point_result}
            )
            # if the number of points in the polygon is equal to the number of
            # points, then the requirements for `.contains_properly` are met
            # for this geometry type.
            df_result = (
                result.groupby("idx").sum().sort_index()
                == result.groupby("idx").count().sort_index()
            )
            point_result = cudf.Series(
                df_result["pip"], index=cudf.RangeIndex(0, len(df_result))
            )
            point_result.name = None
        # Return
        self.op_result = point_result

    def __call__(self) -> cudf.Series:
        return self.op_result

    def preprocess(self, lhs, rhs, align=True):
        # If point.intersects(polygon) is used, reverse lhs and rhs
        def intersects(lhs, rhs, align=True):
            # Preprocessing for intersects called via getattr
            return rhs, lhs

        def contains_properly(lhs, rhs, align=True):
            # Preprocessing for contains_properly called via getattr
            return lhs, rhs

    def contains_properly(self, lhs, points):
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

    def within(lhs, rhs, align=True):
        """An object is said to be within rhs if at least one of its points
        is located in the interior and no points are located in the exterior
        of the rhs."""
        return rhs.contains_properly(lhs, align=align)

    def crosses(lhs, rhs, align=True):
        """Returns a `Series` of `dtype('bool')` with value `True` for each
        aligned geometry that crosses rhs.

        An object is said to cross rhs if its interior intersects the
        interior of the rhs but does not contain it, and the dimension of
        the intersection is less than either."""
        # Crosses requires the use of point_in_polygon but only requires that
        # 1 or more points are within the polygon. This differs from
        # `.contains` which requires all of them.
        return lhs.contains_properly(rhs, align=align)

    def overlaps(lhs, rhs, align=True):
        """Returns True for all aligned geometries that overlap rhs, else
        False.

        Geometries overlap if they have more than one but not all points in
        common, have the same dimension, and the intersection of the
        interiors of the geometries has the same dimension as the geometries
        themselves."""
        # Overlaps has the same requirement as crosses.
        return lhs.contains_properly(rhs, align=align)
