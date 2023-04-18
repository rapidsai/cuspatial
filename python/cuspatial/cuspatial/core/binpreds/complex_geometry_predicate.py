# Copyright (c) 2023, NVIDIA CORPORATION.

from typing import Union

import cupy as cp

import cudf
from cudf.core.dataframe import DataFrame
from cudf.core.series import Series

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    PreprocessorResult,
)
from cuspatial.utils.binpred_utils import (
    _count_results_in_multipoint_geometries,
    _false_series,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
)


class ComplexGeometryPredicate(BinPred):
    def _preprocess_multi(self, lhs, rhs):
        # Breaks down complex geometries into their constituent parts.
        # Passes a tuple o the preprocessed geometries and a tuple of
        # the indices of the points in the original geometry.
        # This is used by the postprocessor to reconstruct the original
        # geometry.
        # Child classes should not implement this method.
        """Flatten any rhs into only its points xy array. This is necessary
        because the basic predicate for contains, point-in-polygon,
        only accepts points.

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
            the right-hand GeoSeries satisfies the requirements of the point-
            in-polygon basic predicate with its corresponding feature in the
            left-hand GeoSeries.
        """
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
        preprocess_result = PreprocessorResult(
            lhs, rhs, final_rhs, point_indices
        )
        return preprocess_result

    def _convert_quadtree_result_from_part_to_polygon_indices(
        self, lhs, point_result
    ):
        """Convert the result of a quadtree contains_properly call from
        part indices to polygon indices.

        Parameters
        ----------
        point_result : cudf.Series
            The result of a quadtree contains_properly call. This result
            contains the `part_index` of the polygon that contains the
            point, not the polygon index.

        Returns
        -------
        cudf.Series
            The result of a quadtree contains_properly call. This result
            contains the `polygon_index` of the polygon that contains the
            point, not the part index.
        """
        # Get the length of each part, map it to indices, and store
        # the result in a dataframe.
        rings_to_parts = cp.array(lhs.polygons.part_offset)
        part_sizes = rings_to_parts[1:] - rings_to_parts[:-1]
        parts_map = cudf.Series(
            cp.arange(len(part_sizes)), name="part_index"
        ).repeat(part_sizes)
        parts_index_mapping_df = parts_map.reset_index(drop=True).reset_index()
        # Map the length of each polygon in a similar fashion, then
        # join them below.
        parts_to_geoms = cp.array(lhs.polygons.geometry_offset)
        geometry_sizes = parts_to_geoms[1:] - parts_to_geoms[:-1]
        geometry_map = cudf.Series(
            cp.arange(len(geometry_sizes)), name="polygon_index"
        ).repeat(geometry_sizes)
        geom_index_mapping_df = geometry_map.reset_index(drop=True)
        geom_index_mapping_df.index.name = "part_index"
        geom_index_mapping_df = geom_index_mapping_df.reset_index()
        # Replace the part index with the polygon index by join
        part_result = parts_index_mapping_df.merge(
            point_result, on="part_index"
        )
        # Replace the polygon index with the row index by join
        return geom_index_mapping_df.merge(part_result, on="part_index")[
            ["polygon_index", "point_index"]
        ]

    def _reindex_allpairs(self, lhs, op_result) -> Union[Series, DataFrame]:
        """Prepare the allpairs result of a contains_properly call as
        the first step of postprocessing.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand side of the binary predicate.
        op_result : ContainsProperlyOpResult
            The result of the contains_properly call.

        Returns
        -------
        cudf.DataFrame

        """
        # Convert the quadtree part indices df into a polygon indices df
        polygon_indices = (
            self._convert_quadtree_result_from_part_to_polygon_indices(
                lhs, op_result.pip_result
            )
        )
        # Because the quadtree contains_properly call returns a list of
        # points that are contained in each part, parts can be duplicated
        # once their index is converted to a polygon index.
        allpairs_result = polygon_indices.drop_duplicates()

        # Replace the polygon index with the original index
        allpairs_result["polygon_index"] = allpairs_result[
            "polygon_index"
        ].replace(Series(lhs.index, index=cp.arange(len(lhs.index))))

        return allpairs_result

    def _postprocess_multi(self, lhs, rhs, preprocessor_result, op_result):
        # Doesn't use op_result, but uses preprocessor_result to
        # reconstruct the original geometry.
        # Child classes should call this method to reconstruct the
        # original geometry.

        # Complex geometry postprocessor
        point_indices = preprocessor_result.point_indices
        allpairs_result = self._reindex_allpairs(lhs, op_result)
        if isinstance(allpairs_result, Series):
            return allpairs_result

        (hits, expected_count,) = _count_results_in_multipoint_geometries(
            point_indices, allpairs_result
        )
        result_df = hits.reset_index().merge(
            expected_count.reset_index(), on="rhs_index"
        )
        result_df["feature_in_polygon"] = (
            result_df["point_index_x"] >= result_df["point_index_y"]
        )
        final_result = _false_series(len(rhs))
        final_result.loc[
            result_df["rhs_index"][result_df["feature_in_polygon"]]
        ] = True
        return final_result
