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

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

#include <tuple>

#include "indexing/construction/detail/utilities.cuh"
#include "utility/z_order.cuh"

namespace cuspatial {

namespace detail {

constexpr uint8_t leaf_indicator = 0;
constexpr uint8_t quad_indicator = 1;
constexpr uint8_t none_indicator = 2;

template <uint8_t NodeType, typename InputIterator, typename OutputIterator>
inline cudf::size_type copy_intersections(InputIterator input_begin,
                                          InputIterator input_end,
                                          OutputIterator output_begin,
                                          cudaStream_t stream)
{
  return thrust::distance(
    output_begin,
    thrust::copy_if(rmm::exec_policy(stream)->on(stream),
                    input_begin,
                    input_end,
                    output_begin,
                    [] __device__(auto const &t) { return thrust::get<1>(t) == NodeType; }));
}

template <uint8_t NodeType, typename InputIterator, typename OutputIterator>
inline cudf::size_type remove_complements(InputIterator input_begin,
                                          InputIterator input_end,
                                          OutputIterator output_begin,
                                          cudaStream_t stream)
{
  return thrust::distance(
    output_begin,
    thrust::remove_if(rmm::exec_policy(stream)->on(stream),
                      input_begin,
                      input_end,
                      output_begin,
                      [] __device__(auto const &t) { return thrust::get<1>(t) != NodeType; }));
}

template <typename T, typename NodesIter, typename PolysIter>
inline std::tuple<cudf::size_type,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint32_t>,
                  cudf::size_type,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint8_t>,
                  rmm::device_uvector<uint32_t>,
                  rmm::device_uvector<uint32_t>>
find_top_level_intersections(NodesIter nodes,
                             PolysIter polys,
                             cudf::size_type num_pairs,
                             cudf::size_type num_nodes,
                             T bbox_x_min,
                             T bbox_y_min,
                             T scale,
                             cudf::size_type max_depth,
                             cudaStream_t stream)
{
  rmm::device_uvector<uint8_t> quad_node_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> quad_node_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> quad_node_indices(num_pairs, stream);
  rmm::device_uvector<uint32_t> quad_poly_indices(num_pairs, stream);

  auto poly_indices = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                      [=](auto const i) { return i / num_nodes; });

  auto node_indices = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                      [=](auto const i) { return i % num_nodes; });

  auto pairs_iter = make_zip_iterator(node_indices,
                                      thrust::make_permutation_iterator(nodes, node_indices),
                                      poly_indices,
                                      thrust::make_permutation_iterator(polys, poly_indices));

  auto nodes_iter = make_zip_iterator(quad_node_types.begin(),
                                      quad_node_levels.begin(),
                                      quad_node_indices.begin(),
                                      quad_poly_indices.begin());

  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    pairs_iter,
                    pairs_iter + num_pairs,
                    nodes_iter,
                    [=] __device__(auto const &pair) {
                      auto &node_idx = thrust::get<0>(pair);
                      auto &key      = thrust::get<0>(thrust::get<1>(pair));
                      auto &level    = thrust::get<1>(thrust::get<1>(pair));
                      auto &is_quad  = thrust::get<2>(thrust::get<1>(pair));

                      auto &poly_idx   = thrust::get<2>(pair);
                      auto &poly_x_min = thrust::get<0>(thrust::get<3>(pair));
                      auto &poly_y_min = thrust::get<1>(thrust::get<3>(pair));
                      auto &poly_x_max = thrust::get<2>(thrust::get<3>(pair));
                      auto &poly_y_max = thrust::get<3>(thrust::get<3>(pair));

                      auto key_x       = cuspatial::utility::z_order_x(key);
                      auto key_y       = cuspatial::utility::z_order_y(key);
                      auto level_scale = scale * pow(T{2.0}, max_depth - 1 - level);
                      auto quad_x_min  = bbox_x_min + (key_x + 0) * level_scale;
                      auto quad_x_max  = bbox_x_min + (key_x + 1) * level_scale;
                      auto quad_y_min  = bbox_y_min + (key_y + 0) * level_scale;
                      auto quad_y_max  = bbox_y_min + (key_y + 1) * level_scale;

                      if (quad_x_min > poly_x_max || quad_x_max < poly_x_min ||
                          quad_y_min > poly_y_max || quad_y_max < poly_y_min) {
                        // if no overlap, return type = none_indicator
                        return thrust::make_tuple(none_indicator, level, node_idx, poly_idx);
                      }
                      // otherwise return type = 0 (leaf) or 1 (quad)
                      return thrust::make_tuple(uint8_t{is_quad}, level, node_idx, poly_idx);
                    });

  rmm::device_uvector<uint8_t> leaf_node_types(num_pairs, stream);
  rmm::device_uvector<uint8_t> leaf_node_levels(num_pairs, stream);
  rmm::device_uvector<uint32_t> leaf_node_indices(num_pairs, stream);
  rmm::device_uvector<uint32_t> leaf_poly_indices(num_pairs, stream);

  auto num_leaf_pairs =
    copy_intersections<leaf_indicator>(nodes_iter,
                                       nodes_iter + num_pairs,
                                       make_zip_iterator(leaf_node_types.begin(),
                                                         leaf_node_levels.begin(),
                                                         leaf_node_indices.begin(),
                                                         leaf_poly_indices.begin()),
                                       stream);

  auto num_quad_pairs =
    remove_complements<quad_indicator>(nodes_iter, nodes_iter + num_pairs, nodes_iter, stream);

  quad_node_types.shrink_to_fit(stream);
  quad_node_levels.shrink_to_fit(stream);
  quad_node_indices.shrink_to_fit(stream);
  quad_poly_indices.shrink_to_fit(stream);

  return std::make_tuple(num_leaf_pairs,
                         std::move(leaf_node_types),
                         std::move(leaf_node_levels),
                         std::move(leaf_node_indices),
                         std::move(leaf_poly_indices),
                         num_quad_pairs,
                         std::move(quad_node_types),
                         std::move(quad_node_levels),
                         std::move(quad_node_indices),
                         std::move(quad_poly_indices));
}

}  // namespace detail
}  // namespace cuspatial
