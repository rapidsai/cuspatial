/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cuspatial/experimental/detail/indexing/construction/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <memory>
#include <tuple>
#include <utility>

/**
 * @brief implementation details for the phase 2 of quadtree construction described in:
 * http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf
 */

namespace cuspatial {
namespace detail {

inline rmm::device_uvector<uint32_t> compute_leaf_positions(
  rmm::device_uvector<bool> const& indicator, int32_t num_valid_nodes, rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint32_t> leaf_pos(num_valid_nodes, stream);
  auto result = thrust::copy_if(rmm::exec_policy(stream),
                                thrust::make_counting_iterator(0),
                                thrust::make_counting_iterator(0) + num_valid_nodes,
                                indicator.begin(),
                                leaf_pos.begin(),
                                !thrust::placeholders::_1);
  // Shrink leaf_pos's underlying device allocation
  leaf_pos.resize(thrust::distance(leaf_pos.begin(), result), stream);
  leaf_pos.shrink_to_fit(stream);
  return leaf_pos;
}

inline rmm::device_uvector<uint32_t> flatten_point_keys(
  rmm::device_uvector<uint32_t> const& quad_keys,
  rmm::device_uvector<uint8_t> const& quad_level,
  rmm::device_uvector<bool> const& indicator,
  int32_t num_valid_nodes,
  int8_t max_depth,
  rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint32_t> flattened_keys(num_valid_nodes, stream);
  auto keys_and_levels =
    thrust::make_zip_iterator(quad_keys.begin(), quad_level.begin(), indicator.begin());
  thrust::transform(rmm::exec_policy(stream),
                    keys_and_levels,
                    keys_and_levels + num_valid_nodes,
                    flattened_keys.begin(),
                    [last_level = max_depth - 1] __device__(auto const& val) {
                      bool is_parent{false};
                      uint32_t key{}, level{};
                      thrust::tie(key, level, is_parent) = val;
                      // if this is a parent node, return max_key. otherwise
                      // compute the key for one level up the tree. Leaf nodes
                      // whose keys are zero will be removed in a subsequent
                      // step
                      return is_parent ? 0xFFFFFFFF : (key << (2 * (last_level - level)));
                    });
  flattened_keys.shrink_to_fit(stream);
  return flattened_keys;
}

/*
 * Adjust quad_point_count and quad_point_pos based on the last level's
 * z-order keys
 */
inline rmm::device_uvector<uint32_t> compute_flattened_first_point_positions(
  rmm::device_uvector<uint32_t> const& quad_keys,
  rmm::device_uvector<uint8_t> const& quad_level,
  rmm::device_uvector<uint32_t>& quad_point_count,
  rmm::device_uvector<bool> const& indicator,
  int32_t num_valid_nodes,
  int8_t max_depth,
  rmm::cuda_stream_view stream)
{
  // Sort initial indices and temporary point counts by the flattened keys
  auto [initial_sort_indices, quad_point_count_tmp] = [&]() {
    auto flattened_keys =
      flatten_point_keys(quad_keys, quad_level, indicator, num_valid_nodes, max_depth, stream);

    rmm::device_uvector<uint32_t> initial_sort_indices(num_valid_nodes, stream);
    thrust::sequence(
      rmm::exec_policy(stream), initial_sort_indices.begin(), initial_sort_indices.end());

    rmm::device_uvector<uint32_t> quad_point_count_tmp(num_valid_nodes, stream);
    thrust::copy(rmm::exec_policy(stream),
                 quad_point_count.begin(),
                 quad_point_count.end(),
                 quad_point_count_tmp.begin());

    // sort indices and temporary point counts
    thrust::stable_sort_by_key(
      rmm::exec_policy(stream),
      flattened_keys.begin(),
      flattened_keys.end(),
      thrust::make_zip_iterator(initial_sort_indices.begin(), quad_point_count_tmp.begin()));

    thrust::remove_if(rmm::exec_policy(stream),
                      quad_point_count_tmp.begin(),
                      quad_point_count_tmp.begin() + num_valid_nodes,
                      quad_point_count_tmp.begin(),
                      thrust::placeholders::_1 == 0);

    initial_sort_indices.shrink_to_fit(stream);
    quad_point_count_tmp.shrink_to_fit(stream);

    return std::make_pair(std::move(initial_sort_indices), std::move(quad_point_count_tmp));
  }();

  auto leaf_offsets = compute_leaf_positions(indicator, num_valid_nodes, stream);

  // Shrink the vector's underlying device allocations.
  // Only the first `num_leaf_nodes` are needed after removal, since
  // copy_if and remove_if should remove the same number of elements.
  quad_point_count_tmp.resize(leaf_offsets.size(), stream);
  quad_point_count_tmp.shrink_to_fit(stream);
  initial_sort_indices.resize(leaf_offsets.size(), stream);
  initial_sort_indices.shrink_to_fit(stream);

  rmm::device_uvector<uint32_t> quad_point_offsets_tmp(leaf_offsets.size(), stream);

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         quad_point_count_tmp.begin(),
                         quad_point_count_tmp.end(),
                         quad_point_offsets_tmp.begin());

  auto counts_and_offsets =
    thrust::make_zip_iterator(quad_point_count_tmp.begin(), quad_point_offsets_tmp.begin());

  thrust::stable_sort_by_key(rmm::exec_policy(stream),
                             initial_sort_indices.begin(),
                             initial_sort_indices.end(),
                             counts_and_offsets);

  rmm::device_uvector<uint32_t> quad_point_offsets(num_valid_nodes, stream);

  thrust::scatter(rmm::exec_policy(stream),
                  counts_and_offsets,
                  counts_and_offsets + leaf_offsets.size(),
                  leaf_offsets.begin(),
                  thrust::make_zip_iterator(quad_point_count.begin(), quad_point_offsets.begin()));

  quad_point_offsets.shrink_to_fit(stream);

  return quad_point_offsets;
}

inline rmm::device_uvector<uint32_t> compute_parent_positions(
  rmm::device_uvector<uint32_t> const& quad_child_count,
  int32_t num_parent_nodes,
  int32_t num_child_nodes,
  rmm::cuda_stream_view stream)
{
  // Compute parent node start positions
  // Wraped in an IIFE so `position_map` is freed on return
  auto parent_pos = [&]() {
    rmm::device_uvector<uint32_t> position_map(num_parent_nodes, stream);
    // line 1 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(rmm::exec_policy(stream),
                           quad_child_count.begin(),
                           quad_child_count.begin() + num_parent_nodes,
                           position_map.begin());
    // line 2 of algorithm in Fig. 5 in ref.
    rmm::device_uvector<uint32_t> parent_pos(num_child_nodes, stream);
    thrust::uninitialized_fill(rmm::exec_policy(stream), parent_pos.begin(), parent_pos.end(), 0);
    thrust::scatter(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(0) + num_parent_nodes,
                    position_map.begin(),
                    parent_pos.begin());
    return parent_pos;
  }();

  // line 3 of algorithm in Fig. 5 in ref.
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         parent_pos.begin(),
                         parent_pos.begin() + num_child_nodes,
                         parent_pos.begin(),
                         thrust::maximum<uint32_t>());

  return parent_pos;
}

/**
 * @brief Remove nodes with fewer than `max_size` number of points, return number of nodes left
 *
 * @param quad_keys
 * @param quad_point_count
 * @param quad_child_count
 * @param quad_levels
 * @param num_parent_nodes
 * @param num_child_nodes
 * @param max_size
 * @param level_1_size
 * @param stream
 * @return std::pair<uint32_t, uint32_t>
 */
inline std::pair<uint32_t, uint32_t> remove_unqualified_quads(
  rmm::device_uvector<uint32_t>& quad_keys,
  rmm::device_uvector<uint32_t>& quad_point_count,
  rmm::device_uvector<uint32_t>& quad_child_count,
  rmm::device_uvector<uint8_t>& quad_levels,
  int32_t num_parent_nodes,
  int32_t num_child_nodes,
  int32_t max_size,
  int32_t level_1_size,
  rmm::cuda_stream_view stream)
{
  // compute parent node start positions
  auto parent_positions =
    compute_parent_positions(quad_child_count, num_parent_nodes, num_child_nodes, stream);
  auto parent_point_counts =
    thrust::make_permutation_iterator(quad_point_count.begin(), parent_positions.begin());

  // Count the number of nodes whose children have fewer points than `max_size`.
  // Start counting nodes at level 2, since children of the root node should not
  // be discarded.
  auto num_invalid_parent_nodes =
    thrust::count_if(rmm::exec_policy(stream),
                     parent_point_counts,
                     parent_point_counts + (num_parent_nodes - level_1_size),
                     // i.e. quad_point_count[parent_pos] <= max_size
                     [max_size] __device__(auto const n) { return n <= max_size; });

  // line 4 of algorithm in Fig. 5 in ref.
  // revision to line 4: copy unnecessary if using permutation_iterator stencil

  // Remove quad nodes with fewer points than `max_size`.
  // Start counting nodes at level 2, since children of the root node should not
  // be discarded.
  // line 5 of algorithm in Fig. 5 in ref.
  auto tree = thrust::make_zip_iterator(quad_keys.begin() + level_1_size,
                                        quad_point_count.begin() + level_1_size,
                                        quad_child_count.begin() + level_1_size,
                                        quad_levels.begin() + level_1_size);

  auto last_valid =
    thrust::remove_if(rmm::exec_policy(stream),
                      tree,
                      tree + num_child_nodes,
                      parent_point_counts,
                      // i.e. quad_point_count[parent_pos] <= max_size
                      [max_size] __device__(auto const n) { return n <= max_size; });

  // add the number of level 1 nodes back in to num_valid_nodes
  auto num_valid_nodes = thrust::distance(tree, last_valid) + level_1_size;

  quad_keys.resize(num_valid_nodes, stream);
  quad_keys.shrink_to_fit(stream);
  quad_point_count.resize(num_valid_nodes, stream);
  quad_point_count.shrink_to_fit(stream);
  quad_child_count.resize(num_valid_nodes, stream);
  quad_child_count.shrink_to_fit(stream);
  quad_levels.resize(num_valid_nodes, stream);
  quad_levels.shrink_to_fit(stream);

  return std::make_pair(num_invalid_parent_nodes, num_valid_nodes);
}

/**
 * @brief Construct the `is_internal_node` vector indicating if a quadrant is a parent or leaf node
 * @param quad_point_count
 * @param num_parent_nodes
 * @param num_valid_nodes
 * @param max_size
 * @param mr
 * @param stream
 * @return rmm::device_uvector<bool>
 */
inline rmm::device_uvector<bool> construct_non_leaf_indicator(
  rmm::device_uvector<uint32_t>& quad_point_count,
  int32_t num_parent_nodes,
  int32_t num_valid_nodes,
  int32_t max_size,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  //
  // Construct the indicator output column
  rmm::device_uvector<bool> is_internal_node(num_valid_nodes, stream, mr);

  // line 6 of algorithm in Fig. 5 in ref.
  thrust::transform(rmm::exec_policy(stream),
                    quad_point_count.begin(),
                    quad_point_count.begin() + num_parent_nodes,
                    is_internal_node.begin(),
                    thrust::placeholders::_1 > max_size);

  // line 7 of algorithm in Fig. 5 in ref.
  thrust::replace_if(rmm::exec_policy(stream),
                     quad_point_count.begin(),
                     quad_point_count.begin() + num_parent_nodes,
                     is_internal_node.begin(),
                     thrust::placeholders::_1,
                     0);

  if (num_valid_nodes > num_parent_nodes) {
    // zero-fill the rest of the indicator column because
    // device_memory_resources aren't required to initialize allocations
    thrust::fill(rmm::exec_policy(stream),
                 is_internal_node.begin() + num_parent_nodes,
                 is_internal_node.end(),
                 0);
  }

  return is_internal_node;
}

}  // namespace detail
}  // namespace cuspatial
