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
#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/memory.h>

#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief The kernel to compute multi-point to multi-linestring distance
 *
 * Each thread computes the distance between a line segment in the linestring and the
 * corresponding multi-point part in the pair. The shortest distance is computed in the
 * output array via an atomic operation.
 *
 * @tparam Cart2dItA Iterator to 2d cartesian coordinates. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Cart2dItB Iterator to 2d cartesian coordinates. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorA Iterator to offsets. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorB Iterator to offsets. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorC Iterator to offsets. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIterator Iterator to output distances. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 *
 * @param[in] point_geometry_offset_first Iterator to the beginning of the range of the multipoint
 * parts
 * @param[in] point_geometry_offset_last Iterator to the end of the range of the multipoint parts
 * @param[in] points_first Iterator to the beginning of the range of the points
 * @param[in] points_last  Iterator to the end of the range of the points
 * @param[in] linestring_geometry_offset_first Iterator to the beginning of the range of the
 * linestring parts
 * @param[in] linestring_geometry_offset_last Iterator to the end of the range of the linestring
 * parts
 * @param[in] linestring_part_offsets_first Iterator to the beginning of the range of the linestring
 * offsets
 * @param[in] linestring_part_offsets_last Iterator to the beginning of the range of the linestring
 * offsets
 * @param[in] linestring_points_first Iterator to the beginning of the range of the linestring
 * points
 * @param[in] linestring_points_last Iterator to the end of the range of the linestring points
 * @param[out] distances Iterator to the output range of shortest distances between pairs.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIterator>
void __global__ pairwise_point_linestring_distance(OffsetIteratorA point_geometry_offset_first,
                                                   OffsetIteratorA point_geometry_offset_last,
                                                   Cart2dItA points_first,
                                                   Cart2dItA points_last,
                                                   OffsetIteratorB linestring_geometry_offset_first,
                                                   OffsetIteratorB linestring_geometry_offset_last,
                                                   OffsetIteratorC linestring_part_offsets_first,
                                                   OffsetIteratorC linestring_part_offsets_last,
                                                   Cart2dItB linestring_points_first,
                                                   Cart2dItB linestring_points_last,
                                                   OutputIterator distances)
{
  using T = iterator_vec_base_type<Cart2dItA>;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < std::distance(linestring_points_first, thrust::prev(linestring_points_last));
       idx += gridDim.x * blockDim.x) {
    // Search from the part offsets array to determine the part idx of current linestring point
    auto linestring_part_offsets_iter = thrust::upper_bound(
      thrust::seq, linestring_part_offsets_first, linestring_part_offsets_last, idx);

    // Pointer to the last point in the linestring, skip iteration.
    // Note that the last point for the last linestring is guarded by the grid-stride loop.
    if (linestring_part_offsets_iter != linestring_part_offsets_last &&
        *linestring_part_offsets_iter - 1 == idx) {
      continue;
    }

    auto part_offsets_idx =
      thrust::distance(linestring_part_offsets_first, thrust::prev(linestring_part_offsets_iter));

    // Search from the linestring geometry offsets array to determine the geometry idx of current
    // linestring point
    auto geometry_offsets_iter = thrust::upper_bound(thrust::seq,
                                                     linestring_geometry_offset_first,
                                                     linestring_geometry_offset_last,
                                                     part_offsets_idx);
    // geometry_idx is also the index to corresponding multipoint in the pair
    auto geometry_idx =
      thrust::distance(linestring_geometry_offset_first, thrust::prev(geometry_offsets_iter));

    // Reduce the minimum distance between different parts of the multi-point.
    vec_2d<T> const a      = linestring_points_first[idx];
    vec_2d<T> const b      = linestring_points_first[idx + 1];
    T min_distance_squared = std::numeric_limits<T>::max();

    for (auto point_idx = point_geometry_offset_first[geometry_idx];
         point_idx < point_geometry_offset_first[geometry_idx + 1];
         point_idx++) {
      vec_2d<T> const c = points_first[point_idx];

      // TODO: reduce redundant computation only related to `a`, `b` in this helper.
      auto const distance_squared = point_to_segment_distance_squared(c, a, b);
      min_distance_squared        = std::min(distance_squared, min_distance_squared);
    }

    atomicMin(&thrust::raw_reference_cast(*(distances + geometry_idx)),
              static_cast<T>(std::sqrt(min_distance_squared)));
  }
}

}  // namespace detail

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_distance(OffsetIteratorA point_geometry_offset_first,
                                            OffsetIteratorA point_geometry_offset_last,
                                            Cart2dItA points_first,
                                            Cart2dItA points_last,
                                            OffsetIteratorB linestring_geometry_offset_first,
                                            OffsetIteratorC linestring_part_offsets_first,
                                            OffsetIteratorC linestring_part_offsets_last,
                                            Cart2dItB linestring_points_first,
                                            Cart2dItB linestring_points_last,
                                            OutputIt distances_first,
                                            rmm::cuda_stream_view stream)
{
  using T = iterator_vec_base_type<Cart2dItA>;

  static_assert(is_same_floating_point<T, iterator_vec_base_type<Cart2dItB>>(),
                "Inputs must have same floating point value type.");

  static_assert(
    is_same<vec_2d<T>, iterator_value_type<Cart2dItA>, iterator_value_type<Cart2dItB>>(),
    "Inputs must be cuspatial::vec_2d");

  auto const num_pairs =
    thrust::distance(point_geometry_offset_first, point_geometry_offset_last) - 1;

  if (num_pairs == 0) { return distances_first; }

  auto const num_linestring_points =
    thrust::distance(linestring_points_first, linestring_points_last);

  thrust::fill_n(
    rmm::exec_policy(stream), distances_first, num_pairs, std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 256;
  std::size_t const num_blocks =
    (num_linestring_points + threads_per_block - 1) / threads_per_block;

  detail::pairwise_point_linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    point_geometry_offset_first,
    point_geometry_offset_last,
    points_first,
    points_last,
    linestring_geometry_offset_first,
    linestring_geometry_offset_first + num_pairs + 1,
    linestring_part_offsets_first,
    linestring_part_offsets_last,
    linestring_points_first,
    linestring_points_last,
    distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return distances_first + num_pairs;
}

}  // namespace cuspatial
