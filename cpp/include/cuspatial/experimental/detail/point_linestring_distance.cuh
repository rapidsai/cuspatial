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
 * @brief The kernel to compute point to linestring distance
 *
 * Each thread computes the distance between a line segment in the linestring and the
 * corresponding point in the pair. The shortest distance is computed in the output
 * array via an atomic operation.
 *
 * @tparam Cart2dItA Iterator to 2d cartesian coordinates. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Cart2dItB Iterator to 2d cartesian coordinates. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIterator Iterator to linestring offsets. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIterator Iterator to output distances. Must meet requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 *
 * @param[in] points_first Iterator to the begin of the range of the points
 * @param[in] points_last  Iterator to the end of the range of the points
 * @param[in] linestring_offsets_begin Iterator to the begin of the range of the linestring offsets
 * @param[in] linestring_offsets_end Iterator to the end of the range of the linestring offsets
 * @param[in] linestring_points_begin Iterator to the begin of the range of the linestring points
 * @param[in] linestring_points_end Iterator to the end of the range of the linestring points
 * @param[out] distances Iterator to the output range of shortest distances between pairs.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename Cart2dItA, typename Cart2dItB, typename OffsetIterator, typename OutputIterator>
void __global__ pairwise_point_linestring_distance(Cart2dItA points_first,
                                                   OffsetIterator linestring_offsets_first,
                                                   OffsetIterator linestring_offsets_last,
                                                   Cart2dItB linestring_points_first,
                                                   Cart2dItB linestring_points_last,
                                                   OutputIterator distances)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  auto const idx = threadIdx.x + blockIdx.x * blockDim.x;
  // Pointer to the last point in linestring array.
  if (linestring_points_first + idx >= thrust::prev(linestring_points_last)) { return; }

  auto offsets_iter =
    thrust::upper_bound(thrust::seq, linestring_offsets_first, linestring_offsets_last, idx);
  // Pointer to the last point in the linestring.
  if (*offsets_iter - 1 == idx) { return; }

  auto pair_idx = thrust::distance(linestring_offsets_first, thrust::prev(offsets_iter));
  cartesian_2d<T> const a  = linestring_points_first[idx];
  cartesian_2d<T> const b  = linestring_points_first[idx + 1];
  cartesian_2d<T> const c  = points_first[pair_idx];

  auto const distance_squared = point_to_segment_distance_squared(c, a, b);

  atomicMin(&thrust::raw_reference_cast(*(distances + pair_idx)),
            static_cast<T>(std::sqrt(distance_squared)));
}

}  // namespace detail

template <class Cart2dItA, class Cart2dItB, class OffsetIterator, class OutputIt>
void pairwise_point_linestring_distance(Cart2dItA points_first,
                                        Cart2dItA points_last,
                                        OffsetIterator linestring_offsets_first,
                                        Cart2dItB linestring_points_first,
                                        Cart2dItB linestring_points_last,
                                        OutputIt distances_first,
                                        rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  static_assert(
    detail::is_floating_point<T,
                              typename std::iterator_traits<Cart2dItB>::value_type::value_type,
                              typename std::iterator_traits<OutputIt>::value_type>(),
    "Inputs and output must have floating point value type.");

  static_assert(detail::is_same<T,
                                typename std::iterator_traits<Cart2dItB>::value_type::value_type,
                                typename std::iterator_traits<OutputIt>::value_type>(),
                "Inputs and output must have the same value type.");

  static_assert(detail::is_same<cartesian_2d<T>,
                                typename std::iterator_traits<Cart2dItA>::value_type,
                                typename std::iterator_traits<Cart2dItB>::value_type>(),
                "Inputs must be cuspatial::cartesian_2d");

  auto const num_pairs = thrust::distance(points_first, points_last);

  if (num_pairs == 0) { return; }

  auto const num_linestring_points =
    thrust::distance(linestring_points_first, linestring_points_last);

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + num_pairs,
               std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 64;
  std::size_t const num_blocks =
    (num_linestring_points + threads_per_block - 1) / threads_per_block;

  detail::pairwise_point_linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    points_first,
    linestring_offsets_first,
    linestring_offsets_first + num_pairs,
    linestring_points_first,
    linestring_points_first + num_linestring_points,
    distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
}

}  // namespace cuspatial
