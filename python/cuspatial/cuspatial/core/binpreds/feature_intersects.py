# Copyright (c) 2023, NVIDIA CORPORATION.

from cuspatial.core.binpreds.feature_contains import RootContains
from cuspatial.utils.column_utils import contains_only_polygons


class RootIntersects(RootContains):
    """Base class for binary predicates that are defined in terms of a
    root-level binary predicate. For example, a Point-Point Intersects
    predicate is defined in terms of a Point-Point Intersects predicate.
    """

    def _preprocess(self, lhs, rhs):
        if contains_only_polygons(rhs):
            (self.lhs, self.rhs) = (rhs, lhs)
        return super()._preprocess(rhs, lhs)

    def _postprocess(self, lhs, rhs, point_indices, point_result):
        # Same as contains_properly, but we need to check that the
        # dimensions are the same.
        contains_result = super()._postprocess(
            lhs, rhs, point_indices, point_result
        )
        return contains_result
