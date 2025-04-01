# Copyright (c) 2024-2025, NVIDIA CORPORATION

import numpy as np

import cudf
from cudf.core.column import ColumnBase, as_column

from cuspatial._lib.distance import (
    pairwise_linestring_distance,
    pairwise_linestring_polygon_distance,
    pairwise_point_distance,
    pairwise_point_linestring_distance,
    pairwise_point_polygon_distance,
    pairwise_polygon_distance,
)
from cuspatial._lib.types import CollectionType
from cuspatial.core._column.geometa import Feature_Enum
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_points,
    contains_only_polygons,
)

# Maps from type combinations to a tuple of (function, reverse,
# point_collection_types).
#
# If reverse is True, the arguments need to be swapped.
# Due to the way the functions are written, certain combinations of types
# requires that the arguments be swapped. For example,
# `point_linestring_distance` requires that the first argument be a point and
# the second argument be a linestring. In this case, when lhs is a linestring
# and rhs is a point, the arguments need to be swapped. The results holds true
# thanks to that cartesian distance is symmetric.
#
# `point_collection_types` is a tuple of the types of the point column type.
# For example, if the first argument is a `MultiPoint` and the second is a
# `Point`, then the `point_collection_types` is (`CollectionType.MULTI`,
# `CollectionType.SINGLE`). They are only needed for point/multipoint columns,
# because the cython APIs are designed to handle both point and multipoint
# columns based on their collection types.

type_to_func = {
    (Feature_Enum.POINT, Feature_Enum.POINT): (
        pairwise_point_distance,
        False,
        (CollectionType.SINGLE, CollectionType.SINGLE),
    ),
    (Feature_Enum.POINT, Feature_Enum.MULTIPOINT): (
        pairwise_point_distance,
        False,
        (CollectionType.SINGLE, CollectionType.MULTI),
    ),
    (Feature_Enum.POINT, Feature_Enum.LINESTRING): (
        pairwise_point_linestring_distance,
        False,
        (CollectionType.SINGLE,),
    ),
    (Feature_Enum.POINT, Feature_Enum.POLYGON): (
        pairwise_point_polygon_distance,
        False,
        (CollectionType.SINGLE,),
    ),
    (Feature_Enum.LINESTRING, Feature_Enum.POINT): (
        pairwise_point_linestring_distance,
        True,
        (CollectionType.SINGLE,),
    ),
    (Feature_Enum.LINESTRING, Feature_Enum.MULTIPOINT): (
        pairwise_point_linestring_distance,
        True,
        (CollectionType.MULTI,),
    ),
    (Feature_Enum.LINESTRING, Feature_Enum.LINESTRING): (
        pairwise_linestring_distance,
        False,
        (),
    ),
    (Feature_Enum.LINESTRING, Feature_Enum.POLYGON): (
        pairwise_linestring_polygon_distance,
        False,
        (),
    ),
    (Feature_Enum.POLYGON, Feature_Enum.POINT): (
        pairwise_point_polygon_distance,
        True,
        (CollectionType.SINGLE,),
    ),
    (Feature_Enum.POLYGON, Feature_Enum.MULTIPOINT): (
        pairwise_point_polygon_distance,
        True,
        (CollectionType.MULTI,),
    ),
    (Feature_Enum.POLYGON, Feature_Enum.LINESTRING): (
        pairwise_linestring_polygon_distance,
        True,
        (),
    ),
    (Feature_Enum.POLYGON, Feature_Enum.POLYGON): (
        pairwise_polygon_distance,
        False,
        (),
    ),
    (Feature_Enum.MULTIPOINT, Feature_Enum.POINT): (
        pairwise_point_distance,
        False,
        (CollectionType.MULTI, CollectionType.SINGLE),
    ),
    (Feature_Enum.MULTIPOINT, Feature_Enum.MULTIPOINT): (
        pairwise_point_distance,
        False,
        (CollectionType.MULTI, CollectionType.MULTI),
    ),
    (Feature_Enum.MULTIPOINT, Feature_Enum.LINESTRING): (
        pairwise_point_linestring_distance,
        False,
        (CollectionType.MULTI,),
    ),
    (Feature_Enum.MULTIPOINT, Feature_Enum.POLYGON): (
        pairwise_point_polygon_distance,
        False,
        (CollectionType.MULTI,),
    ),
}


class DistanceDispatch:
    """Dispatches distance operations between two GeoSeries"""

    def __init__(self, lhs, rhs, align):
        if align:
            self._lhs, self._rhs = lhs.align(rhs)
        else:
            self._lhs, self._rhs = lhs, rhs

        self._align = align
        self._res_index = lhs.index
        self._non_null_mask = self._lhs.notna() & self._rhs.notna()
        self._lhs = self._lhs[self._non_null_mask]
        self._rhs = self._rhs[self._non_null_mask]

        # TODO: This test is expensive, so would be nice if we can cache it
        self._lhs_type = self._determine_series_type(self._lhs)
        self._rhs_type = self._determine_series_type(self._rhs)

    def _determine_series_type(self, s):
        """Check single geometry type of `s`."""
        if contains_only_multipoints(s):
            typ = Feature_Enum.MULTIPOINT
        elif contains_only_points(s):
            typ = Feature_Enum.POINT
        elif contains_only_linestrings(s):
            typ = Feature_Enum.LINESTRING
        elif contains_only_polygons(s):
            typ = Feature_Enum.POLYGON
        else:
            raise NotImplementedError(
                "Geoseries with mixed geometry types are not supported"
            )
        return typ

    def _column(self, s, typ):
        """Get column of `s` based on `typ`."""
        if typ == Feature_Enum.POINT:
            return s.points.column()
        elif typ == Feature_Enum.MULTIPOINT:
            return s.multipoints.column()
        elif typ == Feature_Enum.LINESTRING:
            return s.lines.column()
        elif typ == Feature_Enum.POLYGON:
            return s.polygons.column()

    @property
    def _lhs_column(self):
        return self._column(self._lhs, self._lhs_type)

    @property
    def _rhs_column(self):
        return self._column(self._rhs, self._rhs_type)

    def __call__(self):
        func, reverse, collection_types = type_to_func[
            (self._lhs_type, self._rhs_type)
        ]
        if reverse:
            dist = func(
                *collection_types,
                self._rhs_column.to_pylibcudf(mode="read"),
                self._lhs_column.to_pylibcudf(mode="read"),
            )
        else:
            dist = func(
                *collection_types,
                self._lhs_column.to_pylibcudf(mode="read"),
                self._rhs_column.to_pylibcudf(mode="read"),
            )

        # Rows with misaligned indices contains nan. Here we scatter the
        # distance values to the correct indices.
        result = as_column(
            float("nan"),
            length=len(self._res_index),
            dtype=np.dtype(np.float64),
            nan_as_null=False,
        )
        scatter_map = as_column(
            range(len(self._res_index)), dtype=np.dtype(np.int32)
        ).apply_boolean_mask(self._non_null_mask)

        result[scatter_map] = ColumnBase.from_pylibcudf(dist)

        # If `align==False`, geopandas preserves lhs index.
        index = None if self._align else self._res_index

        return cudf.Series._from_column(result, index=index)
