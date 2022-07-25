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

#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/indexing/construction/phase_1.cuh>
#include <cuspatial/experimental/detail/indexing/construction/phase_2.cuh>
#include <cuspatial/experimental/point_quadtree.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cuspatial {

namespace detail {
/**
 * @brief Constructs a complete quad tree
 */
inline point_quadtree make_quad_tree(rmm::device_uvector<uint32_t>& keys,
                                     rmm::device_uvector<uint32_t>& quad_point_count,
                                     rmm::device_uvector<uint32_t>& quad_child_count,
                                     rmm::device_uvector<uint8_t>& levels,
                                     int32_t num_parent_nodes,
                                     uint8_t max_depth,
                                     uint32_t min_size,
                                     int32_t level_1_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  // count the number of child nodes
  auto num_child_nodes = thrust::reduce(rmm::exec_policy(stream),
                                        quad_child_count.begin(),
                                        quad_child_count.begin() + num_parent_nodes);

  int32_t num_valid_nodes{0};
  int32_t num_invalid_parent_nodes{0};

  // prune quadrants with fewer points than required
  // lines 1, 2, 3, 4, and 5 of algorithm in Fig. 5 in ref.
  std::tie(num_invalid_parent_nodes, num_valid_nodes) = remove_unqualified_quads(keys,
                                                                                 quad_point_count,
                                                                                 quad_child_count,
                                                                                 levels,
                                                                                 num_parent_nodes,
                                                                                 num_child_nodes,
                                                                                 min_size,
                                                                                 level_1_size,
                                                                                 stream);

  num_parent_nodes -= num_invalid_parent_nodes;

  // Construct the indicator output column.
  // line 6 and 7 of algorithm in Fig. 5 in ref.
  auto is_quad = construct_non_leaf_indicator(
    quad_point_count, num_parent_nodes, num_valid_nodes, min_size, mr, stream);

  // Construct the offsets output column
  // lines 8, 9, and 10 of algorithm in Fig. 5 in ref.
  auto offsets = [&]() {
    // line 8 of algorithm in Fig. 5 in ref.
    // revision to line 8: adjust quad_point_pos based on last-level z-order
    // code
    auto quad_point_pos = compute_flattened_first_point_positions(
      keys, levels, quad_point_count, is_quad, num_valid_nodes, max_depth, stream);

    rmm::device_uvector<uint32_t> quad_child_pos(num_valid_nodes, stream, mr);
    // line 9 of algorithm in Fig. 5 in ref.
    thrust::replace_if(rmm::exec_policy(stream),
                       quad_child_count.begin(),
                       quad_child_count.begin() + num_valid_nodes,
                       is_quad.begin(),
                       !thrust::placeholders::_1,
                       0);

    // line 10 of algorithm in Fig. 5 in ref.
    thrust::exclusive_scan(rmm::exec_policy(stream),
                           quad_child_count.begin(),
                           quad_child_count.end(),
                           quad_child_pos.begin(),
                           level_1_size);

    auto& offsets = quad_child_pos;
    auto offsets_iter =
      thrust::make_zip_iterator(is_quad.begin(), quad_child_pos.begin(), quad_point_pos.begin());

    // for each value in `is_quad` copy from `quad_child_pos` if true, else
    // `quad_point_pos`
    thrust::transform(rmm::exec_policy(stream),
                      offsets_iter,
                      offsets_iter + num_valid_nodes,
                      offsets.begin(),
                      // return bool ? lhs : rhs
                      [] __device__(auto const& t) {
                        return thrust::get<0>(t) ? thrust::get<1>(t) : thrust::get<2>(t);
                      });

    return std::move(offsets);
  }();

  // Construct the lengths output column
  rmm::device_uvector<uint32_t> lengths(num_valid_nodes, stream, mr);
  // copy `quad_child_count` if `is_quad` is true, otherwise `quad_point_count`
  auto lengths_iter = thrust::make_zip_iterator(is_quad.begin(),  //
                                                quad_child_count.begin(),
                                                quad_point_count.begin());
  thrust::transform(rmm::exec_policy(stream),
                    lengths_iter,
                    lengths_iter + num_valid_nodes,
                    lengths.begin(),
                    // return bool ? lhs : rhs
                    [] __device__(auto const& t) {
                      return thrust::get<0>(t) ? thrust::get<1>(t) : thrust::get<2>(t);
                    });

  // Shrink keys to the number of valid nodes
  keys.resize(num_valid_nodes, stream);
  keys.shrink_to_fit(stream);

  // Shrink levels to the number of valid nodes
  levels.resize(num_valid_nodes, stream);
  levels.shrink_to_fit(stream);

  return {
    std::move(keys),
    std::move(levels),
    std::move(is_quad),
    std::move(lengths),
    std::move(offsets),
  };
}

/**
 * @brief Constructs a leaf-only quadtree
 */
inline point_quadtree make_leaf_tree(rmm::device_uvector<uint32_t>& keys,
                                     rmm::device_uvector<uint32_t>& lengths,
                                     int32_t num_top_quads,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<uint8_t> levels(num_top_quads, stream, mr);
  rmm::device_uvector<uint8_t> is_quad(num_top_quads, stream, mr);
  rmm::device_uvector<uint32_t> offsets(num_top_quads, stream, mr);

  // only keep the front of the keys list
  keys.resize(num_top_quads, stream);
  keys.shrink_to_fit(stream);
  // only keep the front of the lengths list
  lengths.resize(num_top_quads, stream);
  lengths.shrink_to_fit(stream);

  // All leaves are children of the root node (level 0)
  thrust::fill(rmm::exec_policy(stream), levels.begin(), levels.end(), 0);

  // Quad node indicators are false for leaf nodes
  thrust::fill(rmm::exec_policy(stream), is_quad.begin(), is_quad.end(), false);

  // compute offsets from lengths
  thrust::exclusive_scan(rmm::exec_policy(stream), lengths.begin(), lengths.end(), offsets.begin());

  return {
    std::move(keys),
    std::move(levels),
    std::move(is_quad),
    std::move(lengths),
    std::move(offsets),
  };
}

template <class PointIt,
          class Coord = typename std::iterator_traits<PointIt>::value_type::value_type>
inline std::pair<rmm::device_uvector<uint32_t>, point_quadtree> construct_quadtree(
  PointIt points_first,
  PointIt points_last,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  uint8_t max_depth,
  uint32_t min_size,
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  // Construct the full set of non-empty subquadrants starting from the lowest level.
  // Corresponds to "Phase 1" of quadtree construction in ref.
  auto quads = make_full_levels(points_first,
                                points_last,
                                static_cast<Coord>(x_min),
                                static_cast<Coord>(x_max),
                                static_cast<Coord>(y_min),
                                static_cast<Coord>(y_max),
                                static_cast<Coord>(scale),
                                max_depth,
                                min_size,
                                stream,
                                mr);

  auto& point_indices    = std::get<0>(quads);
  auto& quad_keys        = std::get<1>(quads);
  auto& quad_point_count = std::get<2>(quads);
  auto& quad_child_count = std::get<3>(quads);
  auto& quad_levels      = std::get<4>(quads);
  auto& num_top_quads    = std::get<5>(quads);
  auto& num_parent_nodes = std::get<6>(quads);
  auto& level_1_size     = std::get<7>(quads);

  // Optimization: return early if the top level nodes are all leaves
  if (num_parent_nodes <= 0) {
    return std::make_pair(std::move(point_indices),
                          make_leaf_tree(quad_keys, quad_point_count, num_top_quads, stream, mr));
  }

  // Corresponds to "Phase 2" of quadtree construction in ref.
  return std::make_pair(std::move(point_indices),
                        make_quad_tree(quad_keys,
                                       quad_point_count,
                                       quad_child_count,
                                       quad_levels,
                                       num_parent_nodes,
                                       max_depth,
                                       min_size,
                                       level_1_size,
                                       stream,
                                       mr));
}

}  // namespace detail

template <class PointIt, class Coord>
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
  rmm::mr::device_memory_resource* mr,
  rmm::cuda_stream_view stream)
{
  using T = Coord;
  CUSPATIAL_EXPECTS(x_min < x_max && y_min < y_max,
                    "invalid bounding box (x_min, x_max, y_min, y_max)");
  CUSPATIAL_EXPECTS(scale > 0, "scale must be positive");
  CUSPATIAL_EXPECTS(max_depth < 16, "maximum depth must be less than 16");
  auto num_points = thrust::distance(points_first, points_last);
  if (num_points <= 0) {
    return std::make_pair(rmm::device_uvector<uint32_t>(0, stream),
                          point_quadtree{rmm::device_uvector<uint32_t>(0, stream),
                                         rmm::device_uvector<uint8_t>(0, stream),
                                         rmm::device_uvector<uint8_t>(0, stream),
                                         rmm::device_uvector<uint32_t>(0, stream),
                                         rmm::device_uvector<uint32_t>(0, stream)});
  }
  return detail::construct_quadtree(
    points_first, points_last, x_min, x_max, y_min, y_max, scale, max_depth, min_size, mr, stream);
}

}  // namespace cuspatial
