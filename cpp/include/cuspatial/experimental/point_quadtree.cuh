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

#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

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
  // uint32_t vector of quad node keys
  rmm::device_uvector<uint32_t> key;
  // uint8_t vector of quadtree levels
  rmm::device_uvector<uint8_t> level;
  // bool vector indicating whether the node is a parent (true) or leaf (false) node
  rmm::device_uvector<bool> is_internal_node;
  // uint32_t vector for the number of child nodes (if is_internal_node), or number of points
  rmm::device_uvector<uint32_t> length;
  // uint32_t vector for the first child position (if is_internal_node), or first point position
  rmm::device_uvector<uint32_t> offset;
};

/**
 * @brief Construct a quadtree structure from points.
 *
 * @see http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf for details.
 *
 * @note 2D coordinates are converted into a 1D Morton code by dividing each x/y by the `scale`:
 * (`(x - min_x) / scale` and `(y - min_y) / scale`).
 * @note `max_depth` should be less than 16, since Morton codes are represented as `uint32_t`. The
 * eventual number of levels may be less than `max_depth` if the number of points is small or
 * `max_size` is large.
 * @note All intermediate quadtree nodes will have fewer than `max_size` number of points. Leaf
 * nodes are permitted (but not guaranteed) to have >= `max_size` number of points.
 *
 * @param points Iterator of x, y coordinates for each point.
 * @param vertex_1 Vertex of the area of interest bounding box
 * @param vertex_2 Vertex of the area of interest bounding box opposite `vertex_1`
 * @param scale Scale to apply to each x and y distance from min.x and min.y.
 * @param max_depth Maximum quadtree depth.
 * @param max_size Maximum number of points allowed in a node before it's split into 4 leaf nodes.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * All input iterators must have a `value_type` of `cuspatial::vec_2d<T>` (x/y coordinates), and the
 * output iterator must be able to accept for storage values of type `cuspatial::vec_2d<T>`
 * (Cartesian coordinates).
 *
 * @tparam PointIt Iterator over x/y points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam T the floating-point coordinate value type of the input x/y points.
 *
 * @return Pair of UINT32 column of sorted keys to point indices and a point_quadtree
 */
template <class PointIt, class T = typename cuspatial::iterator_value_type<PointIt>>
std::pair<rmm::device_uvector<uint32_t>, point_quadtree> quadtree_on_points(
  PointIt points_first,
  PointIt points_last,
  vec_2d<T> vertex_1,
  vec_2d<T> vertex_2,
  T scale,
  int8_t max_depth,
  int32_t max_size,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_quadtree.cuh>
