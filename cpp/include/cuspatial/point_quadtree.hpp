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
 * @note `scale` is applied to x_min and y_min to convert x and y coodiantes into a Morton code in
 * 2D space.
 * @note `max_depth` should be less than 16 as uint32_t is used for Morton code representation. The
 * actual number of levels may be less than `max_depth` when the number of points is small and/or
 * `min_size` is large.
 * @note All parent quadrants should have fewer than `min_size` number of points. Leaf quadrants are
 * permited to have more than `min_size` points.
 *
 * @param x Column of x-coordinates for each point.
 * @param y Column of y-coordinates for each point.
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param x_max The upper-right x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param y_max The upper-right y-coordinate of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth.
 * @param min_size Minimum number of points for a non-leaf quadtree node.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @return Pair of INT32 column of sorted keys to point indices, and cudf table with five
 * columns for a complete quadtree:
 *   *     key - INT32 column of quad node keys
 *   *   level - INT8 column of quadtree levels
 *   * is_node - BOOL8 column indicating whether the node is a leaf or not
 *   *  length - INT32 column for the number of child nodes (if is_node), or number of points
 *   *  offset - INT32 column for the first child position (if is_node), or first point position
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::experimental::table>>
quadtree_on_points(cudf::column_view const& x,
                   cudf::column_view const& y,
                   double const x_min,
                   double const x_max,
                   double const y_min,
                   double const y_max,
                   double const scale,
                   cudf::size_type const max_depth,
                   cudf::size_type const min_size,
                   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
