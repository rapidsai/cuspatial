# Copyright (c) 2023, NVIDIA CORPORATION.

import cudf

from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.utils.column_utils import contains_only_polygons


class RootWithin(RootContains):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Within
    predicate is defined in terms of a Point-Point Contains predicate.
    """

    def _preprocess(self, lhs, rhs):
        if contains_only_polygons(rhs):
            (self.lhs, self.rhs) = (rhs, lhs)
        return super()._preprocess(rhs, lhs)

    def _postprocess(self, lhs, rhs, point_indices, point_result):
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
