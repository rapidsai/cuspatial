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
  using T = iterator_vec_base_type<Cart2dItA>;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < std::distance(linestring_points_first, linestring_points_last);
       idx += gridDim.x * blockDim.x) {
    auto offsets_iter =
      thrust::upper_bound(thrust::seq, linestring_offsets_first, linestring_offsets_last, idx);
    // Pointer to the last point in the linestring.
    if (offsets_iter == linestring_offsets_last or *offsets_iter - 1 == idx) { continue; }

    auto point_idx = thrust::distance(linestring_offsets_first, thrust::prev(offsets_iter));
    cartesian_2d<T> const a = linestring_points_first[idx];
    cartesian_2d<T> const b = linestring_points_first[idx + 1];
    cartesian_2d<T> const c = points_first[point_idx];

    auto const distance_squared = point_to_segment_distance_squared(c, a, b);

    atomicMin(&thrust::raw_reference_cast(*(distances + point_idx)),
              static_cast<T>(std::sqrt(distance_squared)));
  }
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
  using T = detail::iterator_vec_base_type<Cart2dItA>;

  static_assert(detail::is_same_floating_point<T,
                                               detail::iterator_vec_base_type<Cart2dItB>,
                                               detail::iterator_value_type<OutputIt>>(),
                "Inputs and output must have same floating point value type.");

  static_assert(detail::is_same<cartesian_2d<T>,
                                detail::iterator_value_type<Cart2dItA>,
                                detail::iterator_value_type<Cart2dItB>>(),
                "Inputs must be cuspatial::cartesian_2d");

  auto const num_pairs = thrust::distance(points_first, points_last);

  if (num_pairs == 0) { return; }

  auto const num_linestring_points =
    thrust::distance(linestring_points_first, linestring_points_last);

  thrust::fill_n(
    rmm::exec_policy(stream), distances_first, num_pairs, std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 256;
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
