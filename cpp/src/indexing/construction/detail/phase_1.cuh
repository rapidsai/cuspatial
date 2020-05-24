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

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "indexing/construction/detail/utilities.cuh"
#include "utility/z_order.cuh"

/**
 * @brief implementation details for the phase 1 of quadtree construction described in:
 * http://www.adms-conf.org/2019-camera-ready/zhang_adms19.pdf
 */

namespace cuspatial {
namespace detail {

/**
 * @brief Compute Morton codes (z-order) for each point, and the sorted indices
 * mapping original point index to sorted z-order.
 */
template <typename T>
inline std::pair<rmm::device_vector<uint32_t>, std::unique_ptr<cudf::column>>
compute_point_keys_and_sorted_indices(cudf::column_view const &x,
                                      cudf::column_view const &y,
                                      T x_min,
                                      T x_max,
                                      T y_min,
                                      T y_max,
                                      T scale,
                                      cudf::size_type max_depth,
                                      rmm::mr::device_memory_resource *mr,
                                      cudaStream_t stream)
{
  rmm::device_vector<uint32_t> keys(x.size());
  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    make_zip_iterator(x.begin<T>(), y.begin<T>()),
                    make_zip_iterator(x.begin<T>(), y.begin<T>()) + x.size(),
                    keys.begin(),
                    [=] __device__(auto const &point) {
                      T x, y;
                      thrust::tie(x, y) = point;
                      if (x < x_min || x > x_max || y < y_min || y > y_max) {
                        // If the point is outside the bbox, return a max_level key
                        return static_cast<uint32_t>((1 << (2 * max_depth)) - 1);
                      }
                      return cuspatial::utility::z_order((x - x_min) / scale, (y - y_min) / scale);
                    });

  auto indices = make_fixed_width_column<int32_t>(keys.size(), stream, mr);

  thrust::sequence(rmm::exec_policy(stream)->on(stream),
                   indices->mutable_view().begin<uint32_t>(),
                   indices->mutable_view().end<uint32_t>());

  // Sort the codes and point indices
  thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                             keys.begin(),
                             keys.end(),
                             indices->mutable_view().begin<int32_t>());

  return std::make_pair(std::move(keys), std::move(indices));
}

/**
 * @brief Reduces the keys and values of the current level into keys and values
 * for the parent level. Returns the number of parent level nodes produced by
 * the reduction.
 */
template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryOp>
inline cudf::size_type build_tree_level(InputIterator1 keys_begin,
                                        InputIterator1 keys_end,
                                        InputIterator2 vals_in,
                                        OutputIterator1 keys_out,
                                        OutputIterator2 vals_out,
                                        BinaryOp binary_op,
                                        cudaStream_t stream)
{
  auto result = thrust::reduce_by_key(rmm::exec_policy(stream)->on(stream),
                                      keys_begin,
                                      keys_end,
                                      vals_in,
                                      keys_out,
                                      vals_out,
                                      thrust::equal_to<uint32_t>(),
                                      binary_op);
  return thrust::distance(keys_out, result.first);
}

/**
 * @brief Construct all quadtree nodes for each level from the bottom-up,
 * starting from the leaf levels and working up to the root node.
 */
template <typename KeysIterator, typename ValsIterator>
inline std::tuple<cudf::size_type,
                  cudf::size_type,
                  std::vector<cudf::size_type>,
                  std::vector<cudf::size_type>>
build_tree_levels(cudf::size_type max_depth,
                  cudf::size_type num_top_quads,
                  KeysIterator keys_begin,
                  ValsIterator quad_point_count_begin,
                  ValsIterator quad_child_count_begin,
                  cudaStream_t stream)
{
  // begin/end offsets
  cudf::size_type begin{0};
  cudf::size_type end{num_top_quads};
  std::vector<cudf::size_type> begin_pos(max_depth);
  std::vector<cudf::size_type> end_pos(max_depth);

  // iterator for the parent level's quad node keys
  auto parent_keys = thrust::make_transform_iterator(
    keys_begin, [] __device__(uint32_t const child_key) { return (child_key >> 2); });

  // iterator for the current level's quad node point and child counts
  auto child_nodes = make_zip_iterator(quad_point_count_begin, quad_child_count_begin);

  // iterator for the current level's initial values
  auto child_values =
    make_zip_iterator(quad_point_count_begin, thrust::make_constant_iterator<uint32_t>(1));

  for (cudf::size_type level = max_depth - 1; level >= 0; --level) {
    auto num_full_quads = build_tree_level(parent_keys + begin,
                                           parent_keys + end,
                                           child_values + begin,
                                           keys_begin + end,
                                           child_nodes + end,
                                           tuple_sum<uint32_t>{},
                                           stream);
    end_pos[level]      = end;
    begin_pos[level]    = begin;
    begin               = end;
    end                 = end + num_full_quads;
  }

  return std::make_tuple(
    // count the number of parent nodes (excluding leaf nodes)
    end - num_top_quads - 1,  //
    end,
    std::move(begin_pos),
    std::move(end_pos));
}

/**
 * @brief Reverse the quadtree nodes for easier manipulation (skipping the root
 * node).
 *
 * The `build_tree_levels` function builds the quadtree from the bottom up,
 * placing the leaf nodes at the front of the quadtree vectors, and the root
 * node at the end. This function reverses the order of the levels, so the level
 * just below the root node is at the front, and the leaves are at the end.
 */
inline std::tuple<rmm::device_vector<uint32_t>,
                  rmm::device_vector<uint32_t>,
                  rmm::device_vector<uint32_t>,
                  rmm::device_vector<int8_t>>
reverse_tree_levels(rmm::device_vector<uint32_t> const &quad_keys_in,
                    rmm::device_vector<uint32_t> const &quad_point_count_in,
                    rmm::device_vector<uint32_t> const &quad_child_count_in,
                    std::vector<cudf::size_type> const &begin_pos,
                    std::vector<cudf::size_type> const &end_pos,
                    cudf::size_type max_depth,
                    cudaStream_t stream)
{
  rmm::device_vector<uint32_t> quad_keys(quad_keys_in.size());
  rmm::device_vector<int8_t> quad_levels(quad_keys_in.size());
  rmm::device_vector<uint32_t> quad_point_count(quad_point_count_in.size());
  rmm::device_vector<uint32_t> quad_child_count(quad_child_count_in.size());
  cudf::size_type offset{0};

  for (cudf::size_type level{0}; level < max_depth; ++level) {
    cudf::size_type level_end   = end_pos[level];
    cudf::size_type level_begin = begin_pos[level];
    cudf::size_type num_quads   = level_end - level_begin;
    thrust::fill(rmm::exec_policy(stream)->on(stream),
                 quad_levels.begin() + offset,
                 quad_levels.begin() + offset + num_quads,
                 level);
    thrust::copy(rmm::exec_policy(stream)->on(stream),
                 quad_keys_in.begin() + level_begin,
                 quad_keys_in.begin() + level_end,
                 quad_keys.begin() + offset);
    thrust::copy(rmm::exec_policy(stream)->on(stream),
                 quad_point_count_in.begin() + level_begin,
                 quad_point_count_in.begin() + level_end,
                 quad_point_count.begin() + offset);
    thrust::copy(rmm::exec_policy(stream)->on(stream),
                 quad_child_count_in.begin() + level_begin,
                 quad_child_count_in.begin() + level_end,
                 quad_child_count.begin() + offset);
    offset += num_quads;
  }

  // Shrink vectors' underlying device allocations to reduce peak memory usage
  quad_keys.shrink_to_fit();
  quad_point_count.shrink_to_fit();
  quad_child_count.shrink_to_fit();
  quad_levels.shrink_to_fit();

  return std::make_tuple(std::move(quad_keys),
                         std::move(quad_point_count),
                         std::move(quad_child_count),
                         std::move(quad_levels));
}

/**
 * @brief Construct the full set of non-empty subquadrants from the lowest level
 * leaves all the way up to the root.
 *
 * * If `num_parent_nodes` is <= 0, construct a leaf-only quadtree.
 * * If `num_parent_nodes` is >= 1, swap each level's position return the
 * swapped vectors. The swapped vectors are used to construct the full quadtree.
 * Freeing the pre-swapped vectors early allows RMM to reuse this memory for
 * subsequent temporary allocations during full quadtree construction.
 *
 * @note Originally part of `dispatch_construct_quadtree`. Split out so
 * `quad_keys`, `quad_point_count`, `quad_child_count`, and `quad_level` vectors
 * are freed on return if we end up swapping the levels.
 *
 */
template <typename T>
inline auto make_full_levels(cudf::column_view const &x,
                             cudf::column_view const &y,
                             T x_min,
                             T x_max,
                             T y_min,
                             T y_max,
                             T scale,
                             cudf::size_type max_depth,
                             cudf::size_type min_size,
                             rmm::mr::device_memory_resource *mr,
                             cudaStream_t stream)
{
  // Compute point keys and sort into top-level quadrants
  // (i.e. quads at level `max_depth - 1`)

  // Compute Morton code (z-order) keys for each point
  auto keys_and_indices = compute_point_keys_and_sorted_indices<T>(
    x, y, x_min, x_max, y_min, y_max, scale, max_depth, mr, stream);

  auto &point_keys    = keys_and_indices.first;
  auto &point_indices = keys_and_indices.second;

  rmm::device_vector<uint32_t> quad_keys(x.size());
  rmm::device_vector<uint32_t> quad_point_count(x.size());

  // Construct quadrants at the finest level of detail, i.e. the quadrants
  // furthest from the root. Reduces points with common z-order codes into
  // the same quadrant.
  auto const num_top_quads = build_tree_level(point_keys.begin(),
                                              point_keys.end(),
                                              thrust::make_constant_iterator<uint32_t>(1),
                                              quad_keys.begin(),
                                              quad_point_count.begin(),
                                              thrust::plus<uint32_t>(),
                                              stream);

  // Optimization: return early if all the nodes are top-level leaf children of the root.
  //
  // This allows us to avoid building the "full" quads for each level only to turn around and
  // remove them for having no children. This scenario causes the `thrust::transform` call in
  // `construct_non_leaf_indicator()` to launch on zero elements.
  // if (num_top_quads >= static_cast<cudf::size_type>(quad_keys.size())) {
  //   return std::make_tuple(std::move(point_indices),
  //                          std::move(quad_keys),
  //                          std::move(quad_point_count),
  //                          std::move(rmm::device_vector<uint32_t>(0)),
  //                          std::move(rmm::device_vector<int8_t>(0)),
  //                          num_top_quads,
  //                          0,
  //                          0);
  // }

  // Repurpose the `point_keys` vector now the points have been grouped into the
  // leaf quadrants
  auto &quad_child_count = point_keys;

  //
  // Compute "full" quads for the tree at each level. Starting from the quadrant
  // at the bottom (at the finest level of detail), aggregates the number of
  // points and children in each quadrant per level. The key for each quadrant
  // is each unique key from the level underneath it shifted right two bits
  // (parent_key = child_key >> 2), as illustrated here:
  //
  // ------------------------------------------------------------------|
  // initial point keys (sorted z-order Morton codes): [1, 3, 4, 7, 8] |
  //                                                    ▾  ▾  ▾  ▾  ▾  |
  //              .-------------------------------------`  :  :  :  :  |
  //              :          .-----------------------------`  :  :  :  |
  //              :          :          .---------------------`  :  :  |
  //              :          :          :          .-------------`  :  |
  //              :          :          :          :          .-----`  |
  // ----------|--+----------+----------+----------+----------+--------|
  //     level |        quadtree nodes [key, num children, num points] |
  // ----------|--+----------+----------+----------+----------+--------|
  //           |  ▾          ▾          ▾          ▾          ▾        |
  // (start) 3 | [1, 0, 1]  [3, 0, 1]  [4, 0, 1]  [7, 0, 1]  [8, 0, 1] |
  //           |     ▾          :          ▾          :          :     |
  //           |     `---------.▾          `---------.▾          ▾     |
  //         2 |            [0, 2, 2]             [1, 2, 2]  [2, 1, 1] |
  //           |                ▾                     :          :     |
  //           |                `--------------------.▾          ▾     |
  //         1 |                                  [0, 2, 4]  [1, 1, 1] |
  //           |                                      ▾          :     |
  //           |                                      `---------.▾     |
  //   (end) 0 |                                             [0, 2, 5] |
  //                                                           (root)  |
  //
  auto quads = build_tree_levels(max_depth,
                                 num_top_quads,
                                 quad_keys.begin(),
                                 quad_point_count.begin(),
                                 quad_child_count.begin(),
                                 stream);

  auto const &num_parent_nodes = std::get<0>(quads);
  auto const &quad_tree_size   = std::get<1>(quads);
  auto const &begin_pos        = std::get<2>(quads);
  auto const &end_pos          = std::get<3>(quads);

  // Shrink vectors' underlying device allocations to reduce peak memory usage
  quad_keys.resize(quad_tree_size);
  quad_keys.shrink_to_fit();
  quad_point_count.resize(quad_tree_size);
  quad_point_count.shrink_to_fit();
  quad_child_count.resize(quad_tree_size);
  quad_child_count.shrink_to_fit();

  // Optimization: return early if the top level nodes are all leaves
  if (num_parent_nodes <= 0) {
    return std::make_tuple(std::move(point_indices),
                           std::move(quad_keys),
                           std::move(quad_point_count),
                           std::move(quad_child_count),
                           std::move(rmm::device_vector<int8_t>(quad_keys.size())),
                           num_top_quads,
                           num_parent_nodes,
                           0);
  }

  //
  // Reverse the quadtree nodes for easier manipulation (skipping the root
  // node).
  //
  // The `build_tree_levels` function builds the quadtree from the bottom up,
  // placing the leaf nodes at the front of the quadtree vectors, and the root
  // node at the end. This function reverses the order of the levels, so the
  // level just below the root node is at the front, and the leaves are at the
  // end.
  auto reversed = reverse_tree_levels(
    quad_keys, quad_point_count, quad_child_count, begin_pos, end_pos, max_depth, stream);

  return std::make_tuple(std::move(point_indices),
                         std::move(std::get<0>(reversed)),
                         std::move(std::get<1>(reversed)),
                         std::move(std::get<2>(reversed)),
                         std::move(std::get<3>(reversed)),
                         num_top_quads,
                         num_parent_nodes,
                         end_pos[0] - begin_pos[0]);
}

}  // namespace detail
}  // namespace cuspatial
