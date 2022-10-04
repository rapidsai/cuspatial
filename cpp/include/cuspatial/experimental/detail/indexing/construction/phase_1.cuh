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

#include <cuspatial/detail/utility/z_order.cuh>
#include <cuspatial/experimental/detail/indexing/construction/utilities.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

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
template <class PointIt, class T>
inline std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<uint32_t>>
compute_point_keys_and_sorted_indices(PointIt points_first,
                                      PointIt points_last,
                                      vec_2d<T> min,
                                      vec_2d<T> max,
                                      T scale,
                                      int8_t max_depth,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  auto num_points = thrust::distance(points_first, points_last);
  rmm::device_uvector<uint32_t> keys(num_points, stream);
  thrust::transform(
    rmm::exec_policy(stream),
    points_first,
    points_last,
    keys.begin(),
    [=] __device__(vec_2d<T> const& point) {
      if (point.x < min.x || point.x > max.x || point.y < min.y || point.y > max.y) {
        // If the point is outside the bbox, return a max_level key
        return static_cast<uint32_t>((1 << (2 * max_depth)) - 1);
      }
      return cuspatial::detail::utility::z_order(static_cast<uint16_t>((point.x - min.x) / scale),
                                                 static_cast<uint16_t>((point.y - min.y) / scale));
    });

  rmm::device_uvector<uint32_t> indices(keys.size(), stream, mr);

  thrust::sequence(rmm::exec_policy(stream), indices.begin(), indices.end());

  // Sort the codes and point indices
  thrust::stable_sort_by_key(rmm::exec_policy(stream), keys.begin(), keys.end(), indices.begin());

  return std::make_pair(std::move(keys), std::move(indices));
}

/**
 * @brief Reduces the keys and values of the current level into keys and values
 * for the parent level. Returns the number of parent level nodes produced by
 * the reduction.
 */
template <typename KeyInputIterator,
          typename ValueInputIterator,
          typename KeyOutputIterator,
          typename ValueOutputIterator,
          typename BinaryOp,
          typename IndexT = typename cuspatial::iterator_value_type<KeyInputIterator>>
inline IndexT build_tree_level(KeyInputIterator keys_begin,
                               KeyInputIterator keys_end,
                               ValueInputIterator vals_in,
                               KeyOutputIterator keys_out,
                               ValueOutputIterator vals_out,
                               BinaryOp binary_op,
                               rmm::cuda_stream_view stream)
{
  auto result = thrust::reduce_by_key(rmm::exec_policy(stream),
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
template <typename KeyInputIterator,
          typename ValueInputIterator,
          typename IndexT = typename cuspatial::iterator_value_type<KeyInputIterator>>
inline std::tuple<IndexT, IndexT, std::vector<IndexT>, std::vector<IndexT>> build_tree_levels(
  KeyInputIterator keys_begin,
  ValueInputIterator quad_point_count_begin,
  ValueInputIterator quad_child_count_begin,
  int8_t max_depth,
  IndexT num_top_quads,
  rmm::cuda_stream_view stream)
{
  // begin/end offsets
  IndexT begin{0};
  IndexT end{num_top_quads};
  std::vector<IndexT> begin_pos(max_depth);
  std::vector<IndexT> end_pos(max_depth);

  // iterator for the parent level's quad node keys
  auto parent_keys = thrust::make_transform_iterator(
    keys_begin, [] __device__(uint32_t const child_key) { return (child_key >> 2); });

  // iterator for the current level's quad node point and child counts
  auto child_nodes = thrust::make_zip_iterator(quad_point_count_begin, quad_child_count_begin);

  // iterator for the current level's initial values
  auto child_values =
    thrust::make_zip_iterator(quad_point_count_begin, thrust::make_constant_iterator<uint32_t>(1));

  for (int32_t level = max_depth - 1; level >= 0; --level) {
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
template <typename IndexT>
inline std::tuple<rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint8_t>>
reverse_tree_levels(rmm::device_uvector<uint32_t> const& quad_keys_in,
                    rmm::device_uvector<uint32_t> const& quad_point_count_in,
                    rmm::device_uvector<uint32_t> const& quad_child_count_in,
                    std::vector<IndexT> const& begin_pos,
                    std::vector<IndexT> const& end_pos,
                    int8_t max_depth,
                    rmm::cuda_stream_view stream)
{
  rmm::device_uvector<uint32_t> quad_keys(quad_keys_in.size(), stream);
  rmm::device_uvector<uint8_t> quad_levels(quad_keys_in.size(), stream);
  rmm::device_uvector<uint32_t> quad_point_count(quad_point_count_in.size(), stream);
  rmm::device_uvector<uint32_t> quad_child_count(quad_child_count_in.size(), stream);
  IndexT offset{0};

  for (int32_t level{0}; level < max_depth; ++level) {
    IndexT level_end   = end_pos[level];
    IndexT level_begin = begin_pos[level];
    IndexT num_quads   = level_end - level_begin;
    thrust::fill(rmm::exec_policy(stream),
                 quad_levels.begin() + offset,
                 quad_levels.begin() + offset + num_quads,
                 level);
    thrust::copy(rmm::exec_policy(stream),
                 quad_keys_in.begin() + level_begin,
                 quad_keys_in.begin() + level_end,
                 quad_keys.begin() + offset);
    thrust::copy(rmm::exec_policy(stream),
                 quad_point_count_in.begin() + level_begin,
                 quad_point_count_in.begin() + level_end,
                 quad_point_count.begin() + offset);
    thrust::copy(rmm::exec_policy(stream),
                 quad_child_count_in.begin() + level_begin,
                 quad_child_count_in.begin() + level_end,
                 quad_child_count.begin() + offset);
    offset += num_quads;
  }

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
template <class PointIt, class T>
inline auto make_full_levels(PointIt points_first,
                             PointIt points_last,
                             vec_2d<T> min,
                             vec_2d<T> max,
                             T scale,
                             int8_t max_depth,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  auto num_points = thrust::distance(points_first, points_last);
  // Compute point keys and sort into bottom-level quadrants
  // (i.e. quads at level `max_depth - 1`)

  // Compute Morton code (z-order) keys for each point
  auto [point_keys, point_indices] = compute_point_keys_and_sorted_indices(
    points_first, points_last, min, max, scale, max_depth, stream, mr);

  rmm::device_uvector<uint32_t> quad_keys(num_points, stream);
  rmm::device_uvector<uint32_t> quad_point_count(num_points, stream);

  // Construct quadrants at the finest level of detail, i.e. the quadrants
  // furthest from the root. Reduces points with common z-order codes into
  // the same quadrant.
  auto const num_bottom_quads = build_tree_level(point_keys.begin(),
                                                 point_keys.end(),
                                                 thrust::make_constant_iterator<uint32_t>(1),
                                                 quad_keys.begin(),
                                                 quad_point_count.begin(),
                                                 thrust::plus<uint32_t>(),
                                                 stream);

  // Repurpose the `point_keys` vector now the points have been grouped into the
  // leaf quadrants
  auto& quad_child_count = point_keys;

  quad_keys.resize(num_bottom_quads * (max_depth + 1), stream);
  quad_point_count.resize(num_bottom_quads * (max_depth + 1), stream);
  quad_child_count.resize(num_bottom_quads * (max_depth + 1), stream);

  // Zero out the quad_child_count vector because we're reusing the point_keys vector
  thrust::fill(rmm::exec_policy(stream), quad_child_count.begin(), quad_child_count.end(), 0);

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
  auto quads = build_tree_levels(quad_keys.begin(),
                                 quad_point_count.begin(),
                                 quad_child_count.begin(),
                                 max_depth,
                                 num_bottom_quads,
                                 stream);

  auto const& num_parent_nodes = std::get<0>(quads);
  auto const& quad_tree_size   = std::get<1>(quads);
  auto const& begin_pos        = std::get<2>(quads);
  auto const& end_pos          = std::get<3>(quads);

  // Shrink vectors' underlying device allocations to reduce peak memory usage
  quad_keys.resize(quad_tree_size, stream);
  quad_keys.shrink_to_fit(stream);
  quad_point_count.resize(quad_tree_size, stream);
  quad_point_count.shrink_to_fit(stream);
  quad_child_count.resize(quad_tree_size, stream);
  quad_child_count.shrink_to_fit(stream);

  // Optimization: return early if the top level nodes are all leaves
  if (num_parent_nodes <= 0) {
    return std::make_tuple(std::move(point_indices),
                           std::move(quad_keys),
                           std::move(quad_point_count),
                           std::move(quad_child_count),
                           std::move(rmm::device_uvector<uint8_t>(quad_keys.size(), stream)),
                           num_bottom_quads,
                           num_parent_nodes,
                           uint32_t{0});
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
                         num_bottom_quads,
                         num_parent_nodes,
                         end_pos[0] - begin_pos[0]);
}

}  // namespace detail
}  // namespace cuspatial
