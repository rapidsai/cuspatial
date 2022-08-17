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

#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/traits.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>

#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 */
template <typename Cart2dItA,
          typename Cart2dItB,
          typename OffsetIterator,
          typename OutputItA,
          typename OutputItB>
void __global__
pairwise_point_linestring_nearest_point_kernel(Cart2dItA points_first,
                                               Cart2dItA points_last,
                                               OffsetIterator linestring_offsets_first,
                                               Cart2dItB linestring_points_first,
                                               Cart2dItB linestring_points_last,
                                               OutputItA nearest_points,
                                               OutputItB nearest_linestring_idxes)
{
  using T       = iterator_vec_base_type<Cart2dItA>;
  using Integer = std::iterator_traits<OffsetIterator>::value_type;

  auto num_pairs             = std::distance(points_first, points_last);
  auto num_linestring_points = std::distance(linestring_points_first, linestring_points_last);
  for (auto pair_idx = threadIdx.x + blockIdx.x * blockDim.x; pair_idx < num_pairs;
       pair_idx += gridDim.x * blockDim.x) {
    Integer linestring_points_start = linestring_offsets_first[pair_idx];
    Integer linestring_points_end   = endpoint_index_of_linestring(
        pair_idx, linestring_offsets_first, num_pairs, num_linestring_points);

    T min_distance_squared = std::numeric_limits<T>::max();
    vec_2d<T> nearest_point;
    Integer nearest_linestring_idx;
    for (auto linestring_point_idx = linestring_points_start;
         linestring_point_idx < linestring_points_end;
         linestring_point_idx++) {
      vec_2d<T> c = points_first[pair_idx];
      vec_2d<T> a = linestring_points_first[linestring_point_idx];
      vec_2d<T> b = linestring_points_first[linestring_point_idx + 1];

      auto distance_point_pair = point_to_segment_distance_squared_nearest_point(c, a, b);
      auto distance_squared    = thrust::get<0>(distance_point_pair);
      if (distance_squared < min_distance_squared) {
        min_distance_squared   = distance_squared;
        nearest_point          = thrust::get<1>(distance_point_pair);
        nearest_linestring_idx = linestring_point_idx;
      }
    }

    nearest_points[pair_idx]           = nearest_point;
    nearest_linestring_idxes[pair_idx] = nearest_linestring_idx;
  }
}

}  // namespace detail

template <class Cart2dItA, class Cart2dItB, class OffsetIterator, class OutputItA, class OutputItB>
void pairwise_point_linestring_nearest_point(Cart2dItA points_first,
                                             Cart2dItA points_last,
                                             OffsetIterator linestring_offsets_first,
                                             Cart2dItB linestring_points_first,
                                             Cart2dItB linestring_points_last,
                                             OutputItA nearest_points,
                                             OutputItB nearest_point_linestring_idx,
                                             rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  auto num_pairs = std::distance(points_first, points_last);

  auto constexpr threads_per_block = 256;
  auto num_blocks                  = (num_pairs + threads_per_block - 1) / threads_per_block;

  detail::pairwise_point_linestring_nearest_point_kernel<<<num_blocks,
                                                           threads_per_block,
                                                           0,
                                                           stream.value()>>>(
    points_first,
    points_last,
    linestring_offsets_first,
    linestring_points_first,
    linestring_points_last,
    nearest_points,
    nearest_point_linestring_idx);
}

}  // namespace cuspatial
