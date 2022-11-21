/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial_indexing
 * @{
 */

/**
 * @brief Construct a quadtree structure from points.
 *
 * @see http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf for details.
 *
 * @note `scale` is applied to (x - x_min) and (y - y_min) to convert coordinates into a Morton code
 * in 2D space.
 * @note `max_depth` should be less than 16, since Morton codes are represented as `uint32_t`. The
 * eventual number of levels may be less than `max_depth` if the number of points is small or
 * `max_size` is large.
 * @note All intermediate quadtree nodes will have fewer than `max_size` number of points. Leaf
 * nodes are permitted (but not guaranteed) to have >= `max_size` number of points.
 *
 * @param x Column of x-coordinates for each point.
 * @param y Column of y-coordinates for each point.
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param x_max The upper-right x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param y_max The upper-right y-coordinate of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth.
 * @param max_size Maximum number of points allowed in a node before it's split into 4 leaf nodes.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If the x and y column sizes are different
 *
 * @return Pair of INT32 column of sorted keys to point indices, and cudf table with five
 * columns for a complete quadtree:
 *     key - UINT32 column of quad node keys
 *   level - UINT8 column of quadtree levels
 * is_internal_node - BOOL8 column indicating whether the node is a quad (true) or leaf (false)
 *  length - UINT32 column for the number of child nodes (if is_internal_node), or number of points
 *  offset - UINT32 column for the first child position (if is_internal_node), or first point
 *           position
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
  cudf::size_type max_size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
