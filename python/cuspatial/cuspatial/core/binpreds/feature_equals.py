# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Generic, TypeVar

import cupy as cp

import cudf
from cudf import Series

import cuspatial
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    EqualsOpResult,
    ImpossiblePredicate,
    NotImplementedPredicate,
    PreprocessorResult,
)
from cuspatial.utils.binpred_utils import (
    LineString,
    MultiPoint,
    Point,
    Polygon,
    _false_series,
)

GeoSeries = TypeVar("GeoSeries")


class EqualsPredicateBase(BinPred, Generic[GeoSeries]):
    """Base class for binary predicates that are defined in terms of the equals
    basic predicate.  `EqualsPredicateBase` implements utility functions that
    are used within many equals-related binary predicates.

    Used by:
    (Point, Point)
    (Point, Polygon)
    (LineString, Polygon)
    (Polygon, Point)
    (Polygon, MultiPoint)
    (Polygon, LineString)
    (Polygon, Polygon)
    """

    def _offset_equals(self, lhs, rhs):
        """Compute the pairwise length equality of two offset arrays. Consider
        the following example:

        lhs = [0, 3, 5, 7]
        rhs = [0, 2, 4, 6]

        _offset_equals(lhs, rhs) returns [False, True, True, True]. The first
        element is False because the first object in lhs has 3 points, while
        the first object in rhs has 2 points. The remaining elements are True
        because the remaining objects in lhs and rhs have the same number of
        points.

        Parameters
        ----------
        lhs : cudf.Series
            left-hand-side offset array
        rhs : cudf.Series
            right-hand-side offset array

        Returns
        -------
        cudf.Series
            pairwise length equality
        """
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
        """Sort xy according to bins defined by offset. Consider an xy buffer
        of 20 values and an offset buffer [0, 5]. This means that the first
        multipoint has 5 points and the second multipoint has 5 points. The
        first multipoint is sorted by x/y coordinates and the second
        multipoint is sorted by x/y coordinates. The resultant sorted values
        are stored in the same offset region, or bin, as the original
        unsorted values.

        Parameters
        ----------
        coords : cudf.Series
            interleaved x,y coordinates
        offsets : cudf.Series
            offsets into coords

        Returns
        -------
        cudf.Series
            Coordinates sorted according to the bins defined by offsets.
        """
        result = self._sort_interleaved_points_by_offset(
            coords, offsets, ["object_key", "xy", "xy_key"]
        )
        result.name = None
        return result

    def _sort_multipoints(self, lhs, rhs):
        """Sort the coordinates of the multipoints in the left-hand and
        right-hand GeoSeries.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand GeoSeries.
        rhs : GeoSeries
            The right-hand GeoSeries.

        Returns
        -------
        lhs_result : Tuple
            A tuple containing the sorted left-hand GeoSeries and the
            sorted right-hand GeoSeries.
        """
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
        # Any unmatched type is not equal
        if (lhs.feature_types != rhs.feature_types).any():
            return _false_series(len(lhs))
        return self._compute_predicate(
            lhs, rhs, PreprocessorResult(None, rhs.point_indices)
        )

    def _vertices_equals(self, lhs: Series, rhs: Series) -> Series:
        """Compute the equals relationship between interleaved xy
        coordinate buffers."""
        if not isinstance(lhs, Series):
            raise TypeError("lhs must be a cudf.Series")
        if not isinstance(rhs, Series):
            raise TypeError("rhs must be a cudf.Series")
        length = min(len(lhs), len(rhs))
        a = lhs[:length:2]._column == rhs[:length:2]._column
        b = rhs[1:length:2]._column == lhs[1:length:2]._column
        return Series._from_column(a & b)

    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        """Perform the binary predicate operation on the input GeoSeries.
        The lhs and rhs are `GeoSeries` of points, and the point_indices
        are the indices of the points in the rhs GeoSeries that correspond
        to each feature in the rhs GeoSeries.
        """
        result = self._vertices_equals(lhs.points.xy, rhs.points.xy)
        return self._postprocess(
            lhs, rhs, EqualsOpResult(result, preprocessor_result.point_indices)
        )

    def _postprocess(self, lhs, rhs, op_result):
        """Postprocess the output GeoSeries to combine the resulting
        comparisons into a single boolean value for each feature in the
        rhs GeoSeries.
        """
        return cudf.Series(op_result.result, dtype="bool")


class PolygonComplexEquals(EqualsPredicateBase):
    def _postprocess(self, lhs, rhs, op_result):
        """Postprocess the output GeoSeries to combine the resulting
        comparisons into a single boolean value for each feature in the
        rhs GeoSeries.
        """
        if len(op_result.result) == 0:
            return _false_series(len(lhs))
        result_df = cudf.DataFrame(
            {"idx": op_result.point_indices, "equals": op_result.result}
        )
        gb_idx = result_df.groupby("idx")
        feature_equals_linestring = (
            gb_idx.sum().sort_index() == gb_idx.count().sort_index()
        )["equals"]
        result = _false_series(len(lhs))
        result[
            feature_equals_linestring.index
        ] = feature_equals_linestring.values
        return result


class MultiPointMultiPointEquals(PolygonComplexEquals):
    def _compute_predicate(self, lhs, rhs, point_indices):
        lengths_equal = self._offset_equals(
            lhs.multipoints.geometry_offset, rhs.multipoints.geometry_offset
        )
        (lhs_sorted, rhs_sorted) = self._sort_multipoints(
            lhs[lengths_equal], rhs[lengths_equal]
        )
        result = self._vertices_equals(
            lhs_sorted.multipoints.xy, rhs_sorted.multipoints.xy
        )
        return self._postprocess(
            lhs, rhs, EqualsOpResult(result, rhs_sorted.point_indices)
        )


class LineStringLineStringEquals(PolygonComplexEquals):
    def _compute_predicate(self, lhs, rhs, preprocessor_result):
        """Linestrings can be compared either forward or reversed. We need
        to compare both directions."""
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
        original_point_indices = cudf.Series(
            lhs_lengths_equal.point_indices
        ).replace(cudf.Series(lhs_lengths_equal.index))
        return self._postprocess(
            lhs, rhs, EqualsOpResult(result, original_point_indices)
        )


class LineStringPointEquals(EqualsPredicateBase):
    def _preprocess(self, lhs, rhs):
        """A LineString cannot be equal to a point. So, return False."""
        return _false_series(len(lhs))


class PolygonPolygonEquals(BinPred):
    def _preprocess(self, lhs, rhs):
        """Two polygons are equal if they contain each other."""
        lhs_contains_rhs = lhs.contains(rhs)
        rhs_contains_lhs = rhs.contains(lhs)
        return lhs_contains_rhs & rhs_contains_lhs


"""DispatchDict for Equals operations."""
DispatchDict = {
    (Point, Point): EqualsPredicateBase,
    (Point, MultiPoint): NotImplementedPredicate,
    (Point, LineString): ImpossiblePredicate,
    (Point, Polygon): EqualsPredicateBase,
    (MultiPoint, Point): NotImplementedPredicate,
    (MultiPoint, MultiPoint): MultiPointMultiPointEquals,
    (MultiPoint, LineString): NotImplementedPredicate,
    (MultiPoint, Polygon): NotImplementedPredicate,
    (LineString, Point): LineStringPointEquals,
    (LineString, MultiPoint): NotImplementedPredicate,
    (LineString, LineString): LineStringLineStringEquals,
    (LineString, Polygon): EqualsPredicateBase,
    (Polygon, Point): EqualsPredicateBase,
    (Polygon, MultiPoint): EqualsPredicateBase,
    (Polygon, LineString): EqualsPredicateBase,
    (Polygon, Polygon): PolygonPolygonEquals,
}
