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

#include <cudf/table/table_view.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_uvector.hpp>

#include <utility>

#include <thrust/copy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include "indexing/construction/detail/utilities.cuh"
#include "utility/z_order.cuh"

namespace cuspatial {
namespace detail {

static __device__ uint8_t const leaf_indicator = 0;
static __device__ uint8_t const quad_indicator = 1;
static __device__ uint8_t const none_indicator = 2;

template <typename InputIterator, typename OutputIterator>
inline cudf::size_type copy_leaf_intersections(InputIterator input_begin,
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
                    [] __device__(auto const &t) { return thrust::get<0>(t) == leaf_indicator; }));
}

template <typename InputIterator, typename OutputIterator>
inline cudf::size_type remove_non_quad_intersections(InputIterator input_begin,
                                                     InputIterator input_end,
                                                     OutputIterator output_begin,
                                                     cudaStream_t stream)
{
  return thrust::distance(output_begin,
                          thrust::remove_if(rmm::exec_policy(stream)->on(stream),
                                            input_begin,
                                            input_end,
                                            output_begin,
                                            [] __device__(auto const &t) {
                                              return thrust::get<0>(t) != quad_indicator;
                                            }));
}

template <typename T,
          typename NodeIndicesIterator,
          typename PolyIndicesIterator,
          typename NodePairsIterator,
          typename LeafPairsIterator>
inline std::pair<cudf::size_type, cudf::size_type> find_intersections(
  cudf::table_view const &quadtree,
  cudf::table_view const &poly_bbox,
  NodeIndicesIterator node_indices,
  PolyIndicesIterator poly_indices,
  NodePairsIterator node_pairs,
  LeafPairsIterator leaf_pairs,
  cudf::size_type num_pairs,
  T x_min,
  T y_min,
  T scale,
  int8_t max_depth,
  cudaStream_t stream)
{
  auto nodes_and_polys = make_zip_iterator(
    node_indices,
    thrust::make_permutation_iterator(quadtree.column(0).begin<uint32_t>(), node_indices),  // keys
    thrust::make_permutation_iterator(quadtree.column(1).begin<uint8_t>(), node_indices),  // levels
    thrust::make_permutation_iterator(quadtree.column(2).begin<bool>(), node_indices),  // is_quad
    poly_indices,
    thrust::make_permutation_iterator(poly_bbox.column(0).begin<T>(), poly_indices),   // poly_x_min
    thrust::make_permutation_iterator(poly_bbox.column(1).begin<T>(), poly_indices),   // poly_y_min
    thrust::make_permutation_iterator(poly_bbox.column(2).begin<T>(), poly_indices),   // poly_x_max
    thrust::make_permutation_iterator(poly_bbox.column(3).begin<T>(), poly_indices));  // poly_y_max

  thrust::transform(rmm::exec_policy(stream)->on(stream),
                    nodes_and_polys,
                    nodes_and_polys + num_pairs,
                    node_pairs,
                    [=] __device__(auto const &node_and_poly) {
                      auto &node_index = thrust::get<0>(node_and_poly);
                      auto &key        = thrust::get<1>(node_and_poly);
                      auto &level      = thrust::get<2>(node_and_poly);
                      auto &is_quad    = thrust::get<3>(node_and_poly);
                      auto &poly_index = thrust::get<4>(node_and_poly);
                      auto &poly_x_min = thrust::get<5>(node_and_poly);
                      auto &poly_y_min = thrust::get<6>(node_and_poly);
                      auto &poly_x_max = thrust::get<7>(node_and_poly);
                      auto &poly_y_max = thrust::get<8>(node_and_poly);

                      T key_x       = cuspatial::utility::z_order_x(key);
                      T key_y       = cuspatial::utility::z_order_y(key);
                      T level_scale = scale * pow(T{2.0}, max_depth - 1 - level);
                      T node_x_min  = x_min + (key_x + 0) * level_scale;
                      T node_y_min  = y_min + (key_y + 0) * level_scale;
                      T node_x_max  = x_min + (key_x + 1) * level_scale;
                      T node_y_max  = y_min + (key_y + 1) * level_scale;

                      if ((node_x_min > poly_x_max) || (node_x_max < poly_x_min) ||
                          (node_y_min > poly_y_max) || (node_y_max < poly_y_min)) {
                        // if no overlap, return type = none_indicator
                        return thrust::make_tuple(none_indicator, level, node_index, poly_index);
                      }
                      // otherwise return type = leaf_indicator (0) or quad_indicator (1)
                      return thrust::make_tuple(
                        static_cast<uint8_t>(is_quad), level, node_index, poly_index);
                    });

  auto num_leaves = copy_leaf_intersections(node_pairs, node_pairs + num_pairs, leaf_pairs, stream);

  auto num_parents =
    remove_non_quad_intersections(node_pairs, node_pairs + num_pairs, node_pairs, stream);

  return std::make_pair(num_parents, num_leaves);
}

}  // namespace detail
}  // namespace cuspatial
