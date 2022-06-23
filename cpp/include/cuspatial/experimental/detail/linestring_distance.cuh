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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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
 * Each thread of the kernel computes the distance between a segment in a linestring in pair 1 to a
 * linestring in pair 2. For a segment in pair 1, the linestring index is looked up from the offset
 * array and mapped to the linestring in the pair 2. The segment is then computed with all segments
 * in the corresponding linestring in pair 2. This forms a local minima of the shortest distance,
 * which is then combined with other segment results via an atomic operation to form the globally
 * minimum distance between the linestrings.
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
 * @param[in] linestring1_offsets_begin Iterator to the begin of the range of linestring offsets in
 * pair 1.
 * @param[in] linestring1_offsets_end Iterator to the end of the range of linestring offsets in
 * pair 1.
 * @param[in] linestring1_points_xs_begin Iterator to the begin of the range of x coordinates of
 * points in pair 1.
 * @param[in] linestring1_points_xs_end Iterator to the end of the range of x coordinates of points
 * in pair 1.
 * @param[in] linestring2_offsets_begin Iterator to the begin of the range of linestring offsets
 * in pair 2.
 * @param[in] linestring2_points_xs_begin Iterator to the begin of the range of x coordinates of
 * points in pair 2.
 * @param[in] linestring2_points_xs_end Iterator to the end of the range of x coordinates of points
 * in pair 2.
 * @param[out] distances Iterator to the output range of shortest distances between pairs.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename Cart2dItA, typename Cart2dItB, typename OffsetIterator, typename OutputIterator>
void __global__ pairwise_linestring_distance_kernel(OffsetIterator linestring1_offsets_begin,
                                                    OffsetIterator linestring1_offsets_end,
                                                    Cart2dItA linestring1_points_begin,
                                                    Cart2dItA linestring1_points_end,
                                                    OffsetIterator linestring2_offsets_begin,
                                                    Cart2dItB linestring2_points_begin,
                                                    Cart2dItB linestring2_points_end,
                                                    OutputIterator distances)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  auto const p1_idx = threadIdx.x + blockIdx.x * blockDim.x;
  std::size_t const num_linestrings =
    thrust::distance(linestring1_offsets_begin, linestring1_offsets_end);

  std::size_t const linestring1_num_points =
    thrust::distance(linestring1_points_begin, linestring1_points_end);
  std::size_t const linestring2_num_points =
    thrust::distance(linestring2_points_begin, linestring2_points_end);

  if (p1_idx >= linestring1_num_points) { return; }

  std::size_t const linestring_idx =
    thrust::distance(linestring1_offsets_begin,
                     thrust::upper_bound(
                       thrust::seq, linestring1_offsets_begin, linestring1_offsets_end, p1_idx)) -
    1;

  auto const ls1_end = endpoint_index_of_linestring(
    linestring_idx, linestring1_offsets_begin, num_linestrings, linestring1_num_points);

  if (p1_idx == ls1_end) {
    // Current point is the end point of the line string.
    return;
  }

  auto const ls2_start = *(linestring2_offsets_begin + linestring_idx);
  auto const ls2_end   = endpoint_index_of_linestring(
    linestring_idx, linestring2_offsets_begin, num_linestrings, linestring2_num_points);

  auto const& A = thrust::raw_reference_cast(linestring1_points_begin[p1_idx]);
  auto const& B = thrust::raw_reference_cast(linestring1_points_begin[p1_idx + 1]);

  auto min_squared_distance = std::numeric_limits<T>::max();
  for (auto p2_idx = ls2_start; p2_idx < ls2_end; p2_idx++) {
    auto const& C        = thrust::raw_reference_cast(linestring2_points_begin[p2_idx]);
    auto const& D        = thrust::raw_reference_cast(linestring2_points_begin[p2_idx + 1]);
    min_squared_distance = std::min(min_squared_distance, squared_segment_distance(A, B, C, D));
  }

  atomicMin(&thrust::raw_reference_cast(*(distances + linestring_idx)),
            static_cast<T>(std::sqrt(min_squared_distance)));
}

}  // namespace detail

template <class Cart2dItA, class Cart2dItB, class OffsetIterator, class OutputIt>
void pairwise_linestring_distance(OffsetIterator linestring1_offsets_first,
                                  OffsetIterator linestring1_offsets_last,
                                  Cart2dItA linestring1_points_first,
                                  Cart2dItA linestring1_points_last,
                                  OffsetIterator linestring2_offsets_first,
                                  Cart2dItB linestring2_points_first,
                                  Cart2dItB linestring2_points_last,
                                  OutputIt distances_first,
                                  rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  static_assert(
    detail::is_floating_point<T,
                              typename std::iterator_traits<Cart2dItB>::value_type::value_type,
                              typename std::iterator_traits<OutputIt>::value_type>(),
    "Inputs and output must have floating point value type.");

  static_assert(detail::is_same<T, typename std::iterator_traits<OutputIt>::value_type>(),
                "Inputs and output must have the same value type.");

  static_assert(detail::is_same<cartesian_2d<T>,
                                typename std::iterator_traits<Cart2dItA>::value_type,
                                typename std::iterator_traits<Cart2dItB>::value_type>(),
                "Input types mismatches or input types are not cuspatial::cartesian_2d");

  auto const num_linestring_pairs =
    thrust::distance(linestring1_offsets_first, linestring1_offsets_last);
  auto const num_linestring1_points =
    thrust::distance(linestring1_points_first, linestring1_points_last);
  auto const num_linestring2_points =
    thrust::distance(linestring2_points_first, linestring2_points_last);

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + num_linestring_pairs,
               std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 64;
  std::size_t const num_blocks =
    (num_linestring1_points + threads_per_block - 1) / threads_per_block;

  detail::pairwise_linestring_distance_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    linestring1_offsets_first,
    linestring1_offsets_last,
    linestring1_points_first,
    linestring1_points_last,
    linestring2_offsets_first,
    linestring2_points_first,
    linestring2_points_last,
    distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
}

}  // namespace cuspatial
