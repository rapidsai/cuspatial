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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/detail/join/intersection.cuh>
#include <cuspatial/experimental/detail/join/traversal.cuh>
#include <cuspatial/experimental/point_quadtree.cuh>

#include <rmm/device_uvector.hpp>

#include <iterator>
#include <utility>

namespace cuspatial {

/**
 * @addtogroup spatial_join
 * @{
 */

/**
 * @brief Search a quadtree for polygon or linestring bounding box intersections.
 *
 * @note 2D coordinates are converted into a 1D Morton code by dividing each x/y by the `scale`:
 * (`(x - min_x) / scale` and `(y - min_y) / scale`).
 * @note `max_depth` should be less than 16, since Morton codes are represented as `uint32_t`. The
 * eventual number of levels may be less than `max_depth` if the number of points is small or
 * `max_size` is large.
 *
 * @param keys_first: start quadtree key iterator
 * @param keys_last: end of quadtree key iterator
 * @param levels_first: start quadtree levels iterator
 * @param is_internal_nodes_first: start quadtree is_internal_node iterator
 * @param lengths_first: start quadtree length iterator
 * @param offsets_first: start quadtree offset iterator
 * @param bounding_boxes_first: start bounding boxes iterator
 * @param bounding_boxes_last: end of bounding boxes iterator
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth at which to stop testing for intersections.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If scale is less than or equal to 0
 * @throw cuspatial::logic_error If max_depth is less than 1 or greater than 15
 *
 * @return A pair of UINT32 bounding box and leaf quadrant offset device vectors:
 *   - bbox_offset - indices for each polygon/linestring bbox that intersects with the quadtree.
 *   - quad_offset - indices for each leaf quadrant intersecting with a polygon/linestring bbox.
 */
template <class KeyIterator,
          class LevelIterator,
          class IsInternalIterator,
          class BoundingBoxIterator,
          class T = typename cuspatial::iterator_vec_base_type<BoundingBoxIterator>>
std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<uint32_t>>
join_quadtree_and_bounding_boxes(
  KeyIterator keys_first,
  KeyIterator keys_last,
  LevelIterator levels_first,
  IsInternalIterator is_internal_nodes_first,
  KeyIterator lengths_first,
  KeyIterator offsets_first,
  BoundingBoxIterator bounding_boxes_first,
  BoundingBoxIterator bounding_boxes_last,
  T x_min,
  T y_min,
  T scale,
  int8_t max_depth,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/quadtree_bbox_filtering.cuh>
