/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
// To add a new API to an existing group, just use the @ingroup tag in the API's Doxygen
// comment or @addtogroup in the file, inside the namespace.
// Add a new group by first specifying in the hierarchy below.

/**
 * @defgroup constants Constants
 * @{
 *   @brief Constants used in cuProj APIs
 *   @file constants.hpp
 * @}
 * @defgroup cuproj_types Types
 * @{
 *      @brief Type declarations for cuproj
 *      @file ellipsoid.hpp
 *      @file projection.hpp
 *      @file vec_2d.hpp
 * @}
 * @defgroup projection_factories Projection Factory Functions
 * @{
 *      @brief Factory functions to create coordinate projections
 *
 *      These factories make it easier to create projections from a variety of sources.
 *      @file projection_factories.cuh
 * @}
 * @defgroup projection_parameters Projection Parameters
 * @{
 *      @brief Projection parameters used in cuProj APIs
 *      @file projection_parameters.hpp
 * @}
 * @defgroup operations Operations
 * @{
 *     @brief Projection pipeline operations
 *     @file operation/operation.cuh
 *     @file operation/axis_swap.cuh
 *     @file operation/clamp_angular_coordinates.cuh
 *     @file operation/degrees_to_radians.cuh
 *     @file operation/offset_scale_cartesian_coordinates.cuh
 *     @file operation/transverse_mercator.cuh
 * @}
 * @defgroup exception Exception
 * @{
 *      @brief cuProj exception types
 *      @file error.hpp
 * @}
 */
