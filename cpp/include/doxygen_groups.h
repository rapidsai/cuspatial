/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 * @brief Doxygen group definitions
 */

// This header is only processed by doxygen and does
// not need to be included in any source file.
// Below are the main groups that doxygen uses to build
// the Modules page in the specified order.
//
// To add a new API to an existing group, just use the
// @ingroup tag in the API's doxygen comment.
// Add a new group by first specifying in the hierarchy below.

/**
 * @defgroup cuspatial_constants Constants
 * @{
 *   @brief Constants used in cuspatial APIs
 *   @file constants.hpp
 * @}
 * @defgroup spatial_api Spatial APIs
 * @{
 *      @defgroup coordinate_transform Coordinate Transform
 *      @{
 *          @brief APIs to transform coordinates
 *
 *          This module contains APIs that transforms cartesian and geodesic coordinates.
 *          @file coordinate_transform.hpp
            @file coordinate_transform.cuh
 *      @}
 *      @defgroup distance Distance
 *      @{
 *          @brief Distance computation APIs
 *
 *          @file point_distance.hpp
 *          @file point_distance.cuh
 *          @file point_linestring_distance.hpp
 *          @file point_linestring_distance.cuh
 *          @file linestring_distance.hpp
 *          @file linestring_distance.cuh
 *          @file hausdorff.hpp
 *          @file hausdorff.cuh
 *          @file haversine.hpp
 *          @file haversine.cuh
 *      @}
 *      @defgroup spatial_relationship Spatial Relationship
 *      @{
 *          @brief APIs related to spatial relationship
 *
 *          @file point_in_polygon.hpp
 *          @file point_in_polygon.cuh
 *          @file polygon_bounding_box.hpp
 *          @file polyline_bounding_box.hpp
 *          @file spatial_window.hpp
 *      @}
 *      @defgroup cubic_spline Cubic Spline
 *      @{
 *          @brief APIs related to cubic splines
 *          @file cubic_spline.hpp
 *      @}
 * @}
 * @defgroup trajectory_api Trajectory APIs
 * @{
 *      @brief APIs related to trajectories
 *      @file trajectory.hpp
 * @}
 * @defgroup spatial_indexing Spatial Indexing
 * @{
 *      @brief APIs related to spatial indexing
 *      @file point_quadtree.hpp
 * @}
 * @defgroup spatial_join Spatial Join
 * @{
 *      @brief APIs related to spatial join
 *      @file spatial_join.hpp
 * @}
 * @defgroup cuspatial_types Types
 * @{
 *      @brief Type declarations for cuspatial
 *      @file vec_2d.hpp
 *
 *      @defgroup type_factories Factory Methods
 *      @{
 *          @brief Factory method to create coordinate iterators
 *
 *          CuSpatial functions inside `experimental` folder are header-only and only accepts
 *          input/output iterators on coordinates. These factory functions are convenient ways
 *          to create iterators from data in various format.
 *          @file type_utils.hpp
 *      @}
 * @}
 * @defgroup io I/O
 * @{
 *      @brief APIs for spatial data I/O
 *      @file shapefile_reader.hpp
 * @}
 * @defgroup exception Exception
 * @{
 *      @brief cuSpatial exception types
 *      @file error.hpp
 * @}
 */
