/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cuspatial/detail/utility/z_order.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <utility>

namespace cuspatial {
namespace detail {

static __device__ uint8_t const leaf_indicator = 0;
static __device__ uint8_t const quad_indicator = 1;
static __device__ uint8_t const none_indicator = 2;

template <typename InputIterator, typename OutputIterator>
inline int32_t copy_leaf_intersections(InputIterator input_begin,
                                       InputIterator input_end,
                                       OutputIterator output_begin,
                                       rmm::cuda_stream_view stream)
{
  return thrust::distance(
    output_begin,
    thrust::copy_if(
      rmm::exec_policy(stream), input_begin, input_end, output_begin, [] __device__(auto const& t) {
        return thrust::get<0>(t) == leaf_indicator;
      }));
}

template <typename InputIterator, typename OutputIterator>
inline int32_t remove_non_quad_intersections(InputIterator input_begin,
                                             InputIterator input_end,
                                             OutputIterator output_begin,
                                             rmm::cuda_stream_view stream)
{
  return thrust::distance(
    output_begin,
    thrust::remove_if(
      rmm::exec_policy(stream), input_begin, input_end, output_begin, [] __device__(auto const& t) {
        return thrust::get<0>(t) != quad_indicator;
      }));
}

template <class T,
          class KeyIterator,
          class LevelIterator,
          class IsInternalIterator,
          class BoundingBoxIterator,
          class NodeIndicesIterator,
          class BBoxIndicesIterator,
          class NodePairsIterator,
          class LeafPairsIterator>
inline std::pair<int32_t, int32_t> find_intersections(KeyIterator keys_first,
                                                      LevelIterator levels_first,
                                                      IsInternalIterator is_internal_node_first,
                                                      BoundingBoxIterator bounding_box_first,
                                                      NodeIndicesIterator node_indices,
                                                      BBoxIndicesIterator bbox_indices,
                                                      NodePairsIterator node_pairs,
                                                      LeafPairsIterator leaf_pairs,
                                                      int32_t num_pairs,
                                                      T x_min,
                                                      T y_min,
                                                      T scale,
                                                      int8_t max_depth,
                                                      rmm::cuda_stream_view stream)
{
  auto nodes_first = thrust::make_zip_iterator(keys_first, levels_first, is_internal_node_first);

  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(node_indices, bbox_indices),
    thrust::make_zip_iterator(node_indices, bbox_indices) + num_pairs,
    node_pairs,
    [x_min, y_min, scale, max_depth, nodes = nodes_first, bboxes = bounding_box_first] __device__(
      auto const& node_and_bbox) {
      auto const& node_idx = thrust::get<0>(node_and_bbox);
      auto const& bbox_idx = thrust::get<1>(node_and_bbox);

      auto const& node                = nodes[node_idx];
      uint32_t const& key             = thrust::get<0>(node);
      uint8_t const& level            = thrust::get<1>(node);
      uint8_t const& is_internal_node = thrust::get<2>(node);

      auto const& bbox       = bboxes[bbox_idx];
      auto const& bbox_min   = bbox.v1;
      auto const& bbox_max   = bbox.v2;
      auto const& poly_x_min = bbox_min.x;
      auto const& poly_y_min = bbox_min.y;
      auto const& poly_x_max = bbox_max.x;
      auto const& poly_y_max = bbox_max.y;

      T const key_x       = utility::z_order_x(key);
      T const key_y       = utility::z_order_y(key);
      T const level_scale = scale * (1 << (max_depth - 1 - level));
      T const node_x_min  = x_min + (key_x + 0) * level_scale;
      T const node_y_min  = y_min + (key_y + 0) * level_scale;
      T const node_x_max  = x_min + (key_x + 1) * level_scale;
      T const node_y_max  = y_min + (key_y + 1) * level_scale;

      if ((node_x_min > poly_x_max) || (node_x_max < poly_x_min) || (node_y_min > poly_y_max) ||
          (node_y_max < poly_y_min)) {
        // if no overlap, return type = none_indicator
        return thrust::make_tuple(none_indicator, level, node_idx, bbox_idx);
      }
      // otherwise return type = leaf_indicator (0) or quad_indicator (1)
      return thrust::make_tuple(is_internal_node, level, node_idx, bbox_idx);
    });

  auto num_leaves = copy_leaf_intersections(node_pairs, node_pairs + num_pairs, leaf_pairs, stream);

  auto num_parents =
    remove_non_quad_intersections(node_pairs, node_pairs + num_pairs, node_pairs, stream);

  return std::make_pair(num_parents, num_leaves);
}

}  // namespace detail
}  // namespace cuspatial
