# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Generic, TypeVar

import cupy as cp

import cudf
from cudf import Series

import cuspatial
from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedRoot,
)

GeoSeries = TypeVar("GeoSeries")


class RootEquals(BinPred, Generic[GeoSeries]):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Equals
    predicate is defined in terms of a Point-Point Intersects predicate.
    """

    def __init__(self, **kwargs):
        self.align = kwargs.get("align", False)

    def _false(self):
        return Series(cp.zeros(len(self.lhs), dtype=cp.bool_))

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

    def _sort_multipoints(self, lhs, rhs):
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
        )

    def _reverse_linestrings(self, coords, offsets):
        """Reverse the order of coordinates in a Arrow buffer of coordinates
        and offsets."""
        result = self._sort_interleaved_points_by_offset(
            coords, offsets, ["object_key", "point_key", "xy_key"]
        )
        result.name = None
        return result

    def _preprocess(self, lhs: "GeoSeries", rhs: "GeoSeries"):
        """Convert the input geometry types into buffers of points that can
        then be compared with the equals basic predicate.

        The equals basic predicate is simply the value-equality operation
        on the coordinates of the points in the geometries. This means that
        we can convert any geometry type into a buffer of points and then
        compare the buffers with the equals basic predicate.

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
        type_compare = lhs.feature_types == rhs.feature_types
        # Any unmatched type is not equal
        if (type_compare == False).all():  # noqa: E712
            # Override _op so that it will not be run.
            return self._false()
        return self._op(lhs, rhs, rhs.point_indices)

    def _vertices_equals(self, lhs: Series, rhs: Series):
        """Compute the equals relationship between interleaved xy
        coordinate buffers."""
        if not isinstance(lhs, Series):
            raise TypeError("lhs must be a cudf.Series")
        if not isinstance(rhs, Series):
            raise TypeError("rhs must be a cudf.Series")
        length = min(len(lhs), len(rhs))
        a = lhs[:length:2]._column == rhs[:length:2]._column
        b = rhs[1:length:2]._column == lhs[1:length:2]._column
        return a & b

    def _op(self, lhs, rhs, point_indices):
        """Perform the binary predicate operation on the input GeoSeries.
        The lhs and rhs are `GeoSeries` of points, and the point_indices
        are the indices of the points in the rhs GeoSeries that correspond
        to each feature in the rhs GeoSeries.
        """
        result = self._vertices_equals(lhs.points.xy, rhs.points.xy)
        return self._postprocess(lhs, rhs, point_indices, result)

    # TODO: Function signature here doesn't support the existing function
    # signature.
    # I need to receive, for this predicate, the set of lengths, a cudf Series
    # of booleans that determines if the lengths of linestrings in the lhs and
    # rhs are the same size and the set of indices that correspond to the
    # lengths that are equal.
    def _postprocess(self, lhs, rhs, point_indices, point_result):
        """Postprocess the output GeoSeries to combine the resulting
        comparisons into a single boolean value for each feature in the
        rhs GeoSeries.
        """
        return point_result


class PolygonComplexEquals(RootEquals):
    def _postprocess(self, lhs, rhs, point_indices, point_result):
        """Postprocess the output GeoSeries to combine the resulting
        comparisons into a single boolean value for each feature in the
        rhs GeoSeries.
        """
        if len(point_result) == 0:
            return cudf.Series(cp.tile([False], len(lhs)), dtype="bool")
        result_df = cudf.DataFrame(
            {"idx": point_indices, "equals": point_result}
        )
        gb_idx = result_df.groupby("idx")
        feature_equals_linestring = (
            gb_idx.sum().sort_index() == gb_idx.count().sort_index()
        )["equals"]
        result = cudf.Series(cp.tile(False, len(lhs)), dtype="bool")
        result[
            feature_equals_linestring.index
        ] = feature_equals_linestring.values
        return result


class MultiPointMultiPointEquals(PolygonComplexEquals):
    def _preprocess(self, lhs, rhs):
        # MultiPoints can be compared either forward or
        # reversed. We need to compare both directions.
        (lhs_result, rhs_result) = self._sort_multipoints(lhs, rhs)
        return self._op(lhs_result, rhs_result, rhs_result.point_indices)

    def _op(self, lhs, rhs, point_indices):
        result = self._vertices_equals(lhs.multipoints.xy, rhs.multipoints.xy)
        return self._postprocess(lhs, rhs, point_indices, result)


class LineStringLineStringEquals(PolygonComplexEquals):
    def _op(self, lhs, rhs, point_indices):
        # Linestrings can be compared either forward or
        # reversed. We need to compare both directions.
        lengths_equal = self._offset_equals(
            lhs.lines.part_offset, rhs.lines.part_offset
        )
        lhs_lengths_equal = lhs[lengths_equal]
        rhs_lengths_equal = rhs[lengths_equal]
        lhs_reversed = self._reverse_linestrings(
            lhs_lengths_equal.lines.xy, lhs_lengths_equal.lines.part_offset
        )
        forward_result = self._vertices_equals(
            lhs_lengths_equal.lines.xy, rhs_lengths_equal.lines.xy
        )
        reverse_result = self._vertices_equals(
            lhs_reversed, rhs_lengths_equal.lines.xy
        )
        result = forward_result | reverse_result
        return self._postprocess(
            lhs, rhs, lhs_lengths_equal.point_indices, result
        )


Point = ColumnType.POINT
MultiPoint = ColumnType.MULTIPOINT
LineString = ColumnType.LINESTRING
Polygon = ColumnType.POLYGON


DispatchDict = {
    (Point, Point): RootEquals,
    (Point, MultiPoint): NotImplementedRoot,
    (Point, LineString): NotImplementedRoot,
    (Point, Polygon): RootEquals,
    (MultiPoint, Point): NotImplementedRoot,
    (MultiPoint, MultiPoint): MultiPointMultiPointEquals,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): LineStringLineStringEquals,
    (LineString, Polygon): RootEquals,
    (Polygon, Point): RootEquals,
    (Polygon, MultiPoint): RootEquals,
    (Polygon, LineString): RootEquals,
    (Polygon, Polygon): RootEquals,
}
