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

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cuspatial {

/**
 * @addtogroup spatial_indexing
 * @{
 */

struct point_quadtree {
  // Defaulted destructor
  ~point_quadtree() = default;
  // Defaulted move constructor
  point_quadtree(point_quadtree&&) = default;  ///< Move constructor
  // Defaulted move assignment operator
  point_quadtree& operator=(point_quadtree&&) = default;

  // Delete copy constructor
  point_quadtree(const point_quadtree&) = delete;
  // Delete copy assignment operator
  point_quadtree& operator=(point_quadtree const&) = delete;

  /**
   * @brief UINT32 column of quad node keys
   */
  rmm::device_uvector<uint32_t> key;
  /**
   * @brief UINT8 column of quadtree levels
   */
  rmm::device_uvector<uint8_t> level;
  /**
   * @brief BOOL8 column indicating whether the node is a quad (true) or leaf (false)
   */
  rmm::device_uvector<uint8_t> is_quad;
  /**
   * @brief UINT32 column for the number of child nodes (if is_quad), or number of points
   */
  rmm::device_uvector<uint32_t> length;
  /**
   * @brief UINT32 column for the first child position (if is_quad), or first point position
   */
  rmm::device_uvector<uint32_t> offset;
};

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
 * @param points Iterator of x, y coordinates for each point.
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param x_max The upper-right x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param y_max The upper-right y-coordinate of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth.
 * @param min_size Minimum number of points for a non-leaf quadtree node.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If the x and y column sizes are different
 * @throw cuspatial::logic_error If scale is less than or equal to 0
 * @throw cuspatial::logic_error If min_size is less than or equal to 0
 * @throw cuspatial::logic_error If x_min is greater than x_max
 * @throw cuspatial::logic_error If y_min is greater than y_max
 * @throw cuspatial::logic_error If max_depth is less than 0 or greater than 15
 *
 * @return Pair of UINT32 column of sorted keys to point indices and a complete point_quadtree
 */
template <class PointIt,
          class Coord = typename std::iterator_traits<PointIt>::value_type::value_type>
std::pair<rmm::device_uvector<uint32_t>, point_quadtree> quadtree_on_points(
  PointIt points_first,
  PointIt points_last,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  uint8_t max_depth,
  uint32_t min_size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_quadtree.cuh>
