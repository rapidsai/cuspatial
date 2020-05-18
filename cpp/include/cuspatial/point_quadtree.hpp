/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#pragma once
#include <cudf/types.hpp>
#include <memory>

namespace cuspatial {

/**
 * @brief Construct a quadtree structure from points.
 *
 * @see http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf for details.
 *
 * @param x Column of x coordiantes before[in]/after[out] sorting
 * @param y Column of y coordiantes before[in]/after[out] sorting
 * @param x1 the lower-left x-coordinate of the area of interest bounding box
 * @param y1 the lower-left y-coordinate of the area of interest bounding box
 * @param x2 the upper-right x-coordinate of the area of interest bounding box
 * @param y2 the upper-right y-coordinate of the area of interest bounding box
 * @param scale Grid cell size along both x and y dimensions. Scale is applied
 * to x1 and y1 to convert x/y coodiantes into a Morton code in 2D space
 * @param num_level Largest depth of quadtree nodes. The value should be less
 * than 16 as uint32_t is used for Morton code representation. The actual number
 * of levels may be less than num_level when # of points are small and/or
 * min_size (next) is large.
 * @param min_size Minimum number of points for a non-leaf quadtree node. All
 * non-last-level quadrants should have less than `min_size` points. Last-level
 * quadrants are permited to have more than `min_size` points. `min_size` is
 * typically set to the number of threads in a block used in the two CUDA
 * kernels needed in the spatial refinement step.
 * @param mr The optional resource to use for output device memory allocations
 *
 * @return cudf table with five columns for a complete quadtree:
 *   *     key - cudf::INT32 column of quad node keys
 *   *   level - cudf::INT8 column of quadtree levels
 *   * is_node - cudf::BOOL8 column of bools indicating whether the node is a
 *               leaf or not
 *   *  length - cudf::INT32 column of number of child nodes (if is_node) or
 *               number of points (if not is_node)
 *   *  offset - cudf::INT32 column of first child position (if is_node) or
 *               first point position (if not is_node)
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::experimental::table>>
quadtree_on_points(cudf::column_view const& x,
                   cudf::column_view const& y,
                   double const x1,
                   double const y1,
                   double const x2,
                   double const y2,
                   double const scale,
                   cudf::size_type const num_level,
                   cudf::size_type const min_size,
                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
