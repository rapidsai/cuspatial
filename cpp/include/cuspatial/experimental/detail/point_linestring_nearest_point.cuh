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
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
void __global__
pairwise_point_linestring_nearest_point_kernel(OffsetIteratorA points_geometry_offsets_first,
                                               OffsetIteratorA points_geometry_offsets_last,
                                               Cart2dItA points_first,
                                               Cart2dItA points_last,
                                               OffsetIteratorB linestring_geometry_offsets_first,
                                               OffsetIteratorB linestring_geometry_offsets_last,
                                               OffsetIteratorC linestring_part_offsets_first,
                                               OffsetIteratorC linestring_part_offsets_last,
                                               Cart2dItB linestring_points_first,
                                               Cart2dItB linestring_points_last,
                                               OutputIt output_first)
{
  using T       = iterator_vec_base_type<Cart2dItA>;
  using Integer = iterator_value_type<OffsetIteratorA>;

  auto num_pairs =
    thrust::distance(points_geometry_offsets_first, points_geometry_offsets_last) - 1;
  auto num_linestring_points = thrust::distance(linestring_points_first, linestring_points_last);

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_pairs;
       idx += gridDim.x * blockDim.x) {
    Integer nearest_part_idx;
    Integer nearest_segment_idx;
    vec_2d<T> nearest_point;
    for (auto point_idx = points_geometry_offsets_first[idx];
         point_idx < points_geometry_offsets_first[idx + 1];
         point_idx++) {
      Integer linestring_parts_start = linestring_geometry_offsets_first[idx];
      Integer linestring_parts_end   = linestring_geometry_offsets_first[idx + 1];

      T min_distance_squared = std::numeric_limits<T>::max();

      for (auto part_idx = linestring_parts_start; part_idx < linestring_parts_end; part_idx++) {
        for (auto segment_idx = linestring_part_offsets_first[part_idx];
             segment_idx < linestring_part_offsets_first[part_idx + 1];
             segment_idx++) {
          vec_2d<T> c = points_first[point_idx];
          vec_2d<T> a = linestring_points_first[segment_idx];
          vec_2d<T> b = linestring_points_first[segment_idx + 1];

          auto distance_point_pair = point_to_segment_distance_squared_nearest_point(c, a, b);
          auto distance_squared    = thrust::get<0>(distance_point_pair);
          if (distance_squared < min_distance_squared) {
            min_distance_squared = distance_squared;
            nearest_point        = thrust::get<1>(distance_point_pair);
            nearest_part_idx     = part_idx;
            nearest_segment_idx  = segment_idx;
          }
        }
      }
    }
    output_first[idx] = thrust::make_tuple(nearest_part_idx, nearest_segment_idx, nearest_point);
  }

}  // namespace detail

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_nearest_point(OffsetIteratorA points_geometry_offsets_first,
                                                 OffsetIteratorA points_geometry_offsets_last,
                                                 Cart2dItA points_first,
                                                 Cart2dItA points_last,
                                                 OffsetIteratorB linestring_geometry_offsets_first,
                                                 OffsetIteratorC linestring_part_offsets_first,
                                                 OffsetIteratorC linestring_part_offsets_last,
                                                 Cart2dItB linestring_points_first,
                                                 Cart2dItB linestring_points_last,
                                                 OutputIt output_first,
                                                 rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  auto num_pairs = std::distance(points_geometry_offsets_first, points_geometry_offsets_last) - 1;

  auto constexpr threads_per_block = 256;
  auto num_blocks                  = (num_pairs + threads_per_block - 1) / threads_per_block;

  detail::pairwise_point_linestring_nearest_point_kernel<<<num_blocks,
                                                           threads_per_block,
                                                           0,
                                                           stream.value()>>>(
    points_geometry_offsets_first,
    points_geometry_offsets_last,
    points_first,
    points_last,
    linestring_geometry_offsets_first,
    linestring_geometry_offsets_first + num_pairs + 1,
    linestring_part_offsets_first,
    linestring_part_offsets_last,
    linestring_points_first,
    linestring_points_last,
    output_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return output_first + num_pairs;
}

}  // namespace cuspatial
