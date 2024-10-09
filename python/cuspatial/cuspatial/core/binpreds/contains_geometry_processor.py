# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import cupy as cp

import cudf

from cuspatial.core._column.geocolumn import GeoColumn
from cuspatial.core.binpreds.binpred_interface import (
    BinPred,
    PreprocessorResult,
)
from cuspatial.utils.binpred_utils import (
    _count_results_in_multipoint_geometries,
    _false_series,
    _true_series,
)
from cuspatial.utils.column_utils import (
    contains_only_linestrings,
    contains_only_multipoints,
    contains_only_polygons,
)


class ContainsGeometryProcessor(BinPred):
    def _preprocess_multipoint_rhs(self, lhs, rhs):
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
        result : PreprocessorResult
            A PreprocessorResult object containing the original lhs and rhs,
            the rhs with only its points, and the indices of the points in
            the original rhs.
        """
        # RHS conditioning:
        point_indices = None
        # point in polygon
        if contains_only_linestrings(rhs):
            # condition for linestrings
            geom = rhs.lines
        elif contains_only_polygons(rhs):
            # polygon in polygon
            geom = rhs.polygons
        elif contains_only_multipoints(rhs):
            # mpoint in polygon
            geom = rhs.multipoints
        else:
            # no conditioning is required
            geom = rhs.points
        xy_points = geom.xy

        # Arrange into shape for calling point-in-polygon
        point_indices = geom.point_indices()
        from cuspatial.core.geoseries import GeoSeries

        final_rhs = GeoSeries._from_column(
            GeoColumn._from_points_xy(xy_points._column)
        )
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

    def _reindex_allpairs(self, lhs, op_result) -> cudf.DataFrame:
        """Prepare the allpairs result of a contains_properly call as
        the first step of postprocessing. An allpairs result is reindexed
        by replacing the polygon index with the original index of the
        polygon from the lhs.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand side of the binary predicate.
        op_result : ContainsProperlyOpResult
            The result of the contains_properly call.

        Returns
        -------
        cudf.DataFrame
            A cudf.DataFrame with two columns: `polygon_index` and
            `point_index`. The `polygon_index` column contains the index
            of the polygon from the original lhs that contains the point,
            and the `point_index` column contains the index of the point
            from the preprocessor final_rhs input to point-in-polygon.
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

        # TODO: This is slow and needs optimization
        # Replace the polygon index with the original index
        allpairs_result["polygon_index"] = allpairs_result[
            "polygon_index"
        ].replace(cudf.Series(lhs.index, index=cp.arange(len(lhs.index))))

        return allpairs_result

    def _postprocess_multipoint_rhs(
        self, lhs, rhs, preprocessor_result, op_result, mode
    ):
        """Reconstruct the original geometry from the result of the
        contains_properly call.

        Parameters
        ----------
        lhs : GeoSeries
            The left-hand side of the binary predicate.
        rhs : GeoSeries
            The right-hand side of the binary predicate.
        preprocessor_result : PreprocessorResult
            The result of the preprocessor.
        op_result : ContainsProperlyOpResult
            The result of the contains_properly call.
        mode : str
            The mode of the predicate. Various mode options are available
            to support binary predicates. The mode options are `full`,
            `basic_none`, `basic_any`, and `basic_count`. If the default
            option `full` is specified, `.contains` or .contains_properly`
            will return a boolean series indicating whether each feature
            in the right-hand GeoSeries is contained by the corresponding
            feature in the left-hand GeoSeries. If `basic_none` is
            specified, `.contains` or .contains_properly` returns the
            negation of `basic_any`.`. If `basic_any` is specified, `.contains`
            or `.contains_properly` returns a boolean series indicating
            whether any point in the right-hand GeoSeries is contained by
            the corresponding feature in the left-hand GeoSeries. If the
            `basic_count` option is specified, `.contains` or
            .contains_properly` returns a Series of integers indicating
            the number of points in the right-hand GeoSeries that are
            contained by the corresponding feature in the left-hand GeoSeries.

        Returns
        -------
        cudf.Series
            A boolean series indicating whether each feature in the
            right-hand GeoSeries satisfies the requirements of the point-
            in-polygon basic predicate with its corresponding feature in the
            left-hand GeoSeries."""

        point_indices = preprocessor_result.point_indices
        allpairs_result = self._reindex_allpairs(lhs, op_result)
        if isinstance(allpairs_result, cudf.Series):
            return allpairs_result
        # Hits is the number of calculated points in each polygon
        # Expected count is the sizes of the features in the right-hand
        # GeoSeries
        (hits, expected_count,) = _count_results_in_multipoint_geometries(
            point_indices, allpairs_result
        )
        result_df = hits.reset_index().merge(
            expected_count.reset_index(), on="rhs_index"
        )
        # Handling for the basic predicates
        if mode == "basic_none":
            none_result = _true_series(len(rhs))
            if len(result_df) == 0:
                return none_result
            none_result.loc[result_df["point_index_x"] > 0] = False
            return none_result
        elif mode == "basic_any":
            any_result = _false_series(len(rhs))
            if len(result_df) == 0:
                return any_result
            indexes = result_df["rhs_index"][result_df["point_index_x"] > 0]
            any_result.iloc[indexes] = True
            return any_result
        elif mode == "basic_count":
            count_result = cudf.Series(cp.zeros(len(rhs)), dtype="int32")
            if len(result_df) == 0:
                return count_result
            hits = result_df["point_index_x"]
            hits.index = count_result.iloc[result_df["rhs_index"]].index
            count_result = count_result.astype(hits.dtype)
            count_result.iloc[result_df["rhs_index"]] = hits
            return count_result

        # Handling for full contains (equivalent to basic predicate all)
        # for each input pair i: result[i] =  true iff point[i] is
        # contained in at least one polygon of multipolygon[i].
        result_df["feature_in_polygon"] = (
            result_df["point_index_x"] >= result_df["point_index_y"]
        )
        final_result = _false_series(len(rhs))
        final_result.loc[
            result_df["rhs_index"][result_df["feature_in_polygon"]]
        ] = True
        return final_result

    def _postprocess_points(self, lhs, rhs, preprocessor_result, op_result):
        """Used when the rhs is naturally points. Instead of reconstructing
        the original geometry, this method applies the `point_index` results
        to the original rhs points and returns a boolean series reflecting
        which `point_index`es were found.
        """
        allpairs_result = self._reindex_allpairs(lhs, op_result)
        if self.config.allpairs:
            return allpairs_result

        final_result = _false_series(len(rhs))
        if len(lhs) == len(rhs):
            matches = (
                allpairs_result["polygon_index"]
                == allpairs_result["point_index"]
            )
            polygon_indexes = allpairs_result["polygon_index"][matches]
            final_result.loc[
                preprocessor_result.point_indices[polygon_indexes]
            ] = True
            return final_result
        else:
            final_result.loc[allpairs_result["polygon_index"]] = True
            return final_result
