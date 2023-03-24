# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Generic, TypeVar

import cupy as cp

import cudf

import cuspatial
from cuspatial.core._column.geocolumn import ColumnType
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    NotImplementedRoot,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_points,
    contains_only_polygons,
)

GeoSeries = TypeVar("GeoSeries")


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


class RootEquals(BinPred, Generic[GeoSeries]):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Equals
    predicate is defined in terms of a Point-Point Intersects predicate.
    """

    def __init__(self, lhs: GeoSeries, rhs: GeoSeries, **kwargs):
        self.lhs = lhs
        self.rhs = rhs

    def __call__(self):
        return self._call()

    def _call(self):
        return self._preprocess(self.lhs, self.rhs)

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
                point_indices = self._sort_multipoints(
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
                point_indices = (
                    lhs[lengths_equal],
                    rhs[lengths_equal],
                    lengths_equal,
                )
            else:
                return self._cancel_op(lhs, rhs, lengths_equal)
        elif contains_only_polygons(lhs):
            raise NotImplementedError
        elif contains_only_points(lhs):
            point_indices = type_compare
        return self._op(lhs, rhs, point_indices)

    def _vertices_equals(self, lhs, rhs):
        """Compute the equals relationship between interleaved xy
        coordinate buffers."""
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
    def _postprocess(self, lhs, rhs, lengths_equal, point_result):
        """Postprocess the output GeoSeries to combine the resulting
        comparisons into a single boolean value for each feature in the
        rhs GeoSeries.
        """
        # if point_result is not a Series, preprocessing terminated
        # the results early.
        if isinstance(point_result, cudf.Series):
            op_result = point_result.sort_index()
            lengths_equal[point_result.index] = op_result
            return cudf.Series(lengths_equal)
        indices = lhs.point_indices()
        result_df = cudf.DataFrame(
            {"idx": indices[: len(point_result)], "equals": point_result}
        )
        gb_idx = result_df.groupby("idx")
        result = (gb_idx.sum().sort_index() == gb_idx.count().sort_index())[
            "equals"
        ]
        result.index = lhs.index
        result.index.name = None
        result.name = None
        return result


class PointPointEquals(RootEquals):
    pass


class PolygonPointEquals(RootEquals):
    pass


class PolygonComplexEquals(RootEquals):
    pass


class PolygonMultiPointEquals(PolygonComplexEquals):
    def _op(self, lhs, rhs, point_indices):
        result = self._vertices_equals(lhs.multipoints.xy, rhs.multipoints.xy)
        return self._postprocess(lhs, rhs, point_indices, result)


class PolygonLineStringEquals(PolygonComplexEquals):
    # Linestrings can be compared either forward or
    # reversed. We need to compare both directions.
    def _op(self, lhs, rhs, point_indices):
        lhs_reversed = self._reverse_linestrings(
            lhs.lines.xy, lhs.lines.part_offset
        )
        forward_result = self._vertices_equals(lhs.lines.xy, rhs.lines.xy)
        reverse_result = self._vertices_equals(lhs_reversed, rhs.lines.xy)
        result = forward_result | reverse_result
        return self._postprocess(lhs, rhs, point_indices, result)


class PolygonMultiLineStringEquals(PolygonComplexEquals):
    pass


class PolygonPolygonEquals(PolygonComplexEquals):
    def _op(self, lhs, rhs, point_indices):
        raise NotImplementedError


class PolygonMultiPolygonEquals(PolygonComplexEquals):
    pass


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
    (MultiPoint, MultiPoint): NotImplementedRoot,
    (MultiPoint, LineString): NotImplementedRoot,
    (MultiPoint, Polygon): NotImplementedRoot,
    (LineString, Point): NotImplementedRoot,
    (LineString, MultiPoint): NotImplementedRoot,
    (LineString, LineString): NotImplementedRoot,
    (LineString, Polygon): RootEquals,
    (Polygon, Point): PolygonPointEquals,
    (Polygon, MultiPoint): PolygonMultiLineStringEquals,
    (Polygon, LineString): PolygonLineStringEquals,
    (Polygon, Polygon): RootEquals,
}
