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
 * @note `scale` is applied to (x - x_min) and (y - y_min) to convert coordinates into a Morton code
 * in 2D space.
 * @note `max_depth` should be less than 16, since Morton codes are represented as `uint32_t`. The
 * eventual number of levels may be less than `max_depth` if the number of points is small or
 * `min_size` is large.
 * @note All quadtree nodes should have fewer than `min_size` number of points except leaf
 * quadrants, which are permitted to have more than `min_size` points.
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
 *     key - UINT32 column of quad node keys
 *   level - UINT8 column of quadtree levels
 * is_quad - BOOL8 column indicating whether the node is a leaf or not
 *  length - UINT32 column for the number of child nodes (if is_quad), or number of points
 *  offset - UINT32 column for the first child position (if is_quad), or first point position
 */
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::table>> quadtree_on_points(
  cudf::column_view const& x,
  cudf::column_view const& y,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  int8_t max_depth,
  cudf::size_type min_size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
