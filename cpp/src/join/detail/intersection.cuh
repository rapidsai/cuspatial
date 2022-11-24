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

#include <indexing/construction/detail/utilities.cuh>

#include <cuspatial/detail/utility/z_order.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
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
inline cudf::size_type copy_leaf_intersections(InputIterator input_begin,
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
inline cudf::size_type remove_non_quad_intersections(InputIterator input_begin,
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

template <typename T,
          typename NodeIndicesIterator,
          typename PolyIndicesIterator,
          typename NodePairsIterator,
          typename LeafPairsIterator>
inline std::pair<cudf::size_type, cudf::size_type> find_intersections(
  cudf::table_view const& quadtree,
  cudf::table_view const& poly_bbox,
  NodeIndicesIterator node_indices,
  PolyIndicesIterator poly_indices,
  NodePairsIterator node_pairs,
  LeafPairsIterator leaf_pairs,
  cudf::size_type num_pairs,
  T x_min,
  T y_min,
  T scale,
  int8_t max_depth,
  rmm::cuda_stream_view stream)
{
  auto d_keys             = cudf::column_device_view::create(quadtree.column(0), stream);
  auto d_levels           = cudf::column_device_view::create(quadtree.column(1), stream);
  auto d_is_internal_node = cudf::column_device_view::create(quadtree.column(2), stream);
  auto d_poly_x_min       = cudf::column_device_view::create(poly_bbox.column(0), stream);
  auto d_poly_y_min       = cudf::column_device_view::create(poly_bbox.column(1), stream);
  auto d_poly_x_max       = cudf::column_device_view::create(poly_bbox.column(2), stream);
  auto d_poly_y_max       = cudf::column_device_view::create(poly_bbox.column(3), stream);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_zip_iterator(node_indices, poly_indices),
                    thrust::make_zip_iterator(node_indices, poly_indices) + num_pairs,
                    node_pairs,
                    [x_min,
                     y_min,
                     scale,
                     max_depth,
                     keys             = *d_keys,
                     levels           = *d_levels,
                     is_internal_node = *d_is_internal_node,
                     poly_x_mins      = *d_poly_x_min,
                     poly_y_mins      = *d_poly_y_min,
                     poly_x_maxs      = *d_poly_x_max,
                     poly_y_maxs      = *d_poly_y_max] __device__(auto const& node_and_poly) {
                      auto& node      = thrust::get<0>(node_and_poly);
                      auto& poly      = thrust::get<1>(node_and_poly);
                      auto key        = keys.element<uint32_t>(node);
                      auto level      = levels.element<uint8_t>(node);
                      auto poly_x_min = poly_x_mins.element<T>(poly);
                      auto poly_y_min = poly_y_mins.element<T>(poly);
                      auto poly_x_max = poly_x_maxs.element<T>(poly);
                      auto poly_y_max = poly_y_maxs.element<T>(poly);

                      T key_x       = utility::z_order_x(key);
                      T key_y       = utility::z_order_y(key);
                      T level_scale = scale * (1 << (max_depth - 1 - level));
                      T node_x_min  = x_min + (key_x + 0) * level_scale;
                      T node_y_min  = y_min + (key_y + 0) * level_scale;
                      T node_x_max  = x_min + (key_x + 1) * level_scale;
                      T node_y_max  = y_min + (key_y + 1) * level_scale;

                      if ((node_x_min > poly_x_max) || (node_x_max < poly_x_min) ||
                          (node_y_min > poly_y_max) || (node_y_max < poly_y_min)) {
                        // if no overlap, return type = none_indicator
                        return thrust::make_tuple(none_indicator, level, node, poly);
                      }
                      // otherwise return type = leaf_indicator (0) or quad_indicator (1)
                      return thrust::make_tuple(
                        is_internal_node.element<uint8_t>(node), level, node, poly);
                    });

  auto num_leaves = copy_leaf_intersections(node_pairs, node_pairs + num_pairs, leaf_pairs, stream);

  auto num_parents =
    remove_non_quad_intersections(node_pairs, node_pairs + num_pairs, node_pairs, stream);

  return std::make_pair(num_parents, num_leaves);
}

}  // namespace detail
}  // namespace cuspatial
