/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/optional.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief The kernel to compute linestring to linestring distance
 *
 * Each thread of the kernel computes the distance between a segment in a linestring in pair 1 to a
 * linestring in pair 2. For a segment in pair 1, the linestring index is looked up from the offset
 * array and mapped to the linestring in the pair 2. The segment is then computed with all segments
 * in the corresponding linestring in pair 2. This forms a local minima of the shortest distance,
 * which is then combined with other segment results via an atomic operation to form the global
 * minimum distance between the linestrings.
 *
 * `intersects` is an optional pointer to a boolean range where the `i`th element indicates the
 * `i`th output should be set to 0 and bypass distance computation. This argument is optional, if
 * set to nullopt, no distance computation will be bypassed.
 */
template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
__global__ void linestring_distance(MultiLinestringRange1 multilinestrings1,
                                    MultiLinestringRange2 multilinestrings2,
                                    thrust::optional<uint8_t*> intersects,
                                    OutputIt distances_first)
{
  using T = typename MultiLinestringRange1::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);

    if (intersects.has_value() && intersects.value()[geometry_idx]) {
      distances_first[geometry_idx] = 0;
      continue;
    }

    auto [a, b]            = multilinestrings1.segment(idx);
    T min_distance_squared = std::numeric_limits<T>::max();

    for (auto const& linestring2 : multilinestrings2[geometry_idx]) {
      for (auto [c, d] : linestring2) {
        min_distance_squared = min(min_distance_squared, squared_segment_distance(a, b, c, d));
      }
    }
    atomicMin(&distances_first[geometry_idx], static_cast<T>(sqrt(min_distance_squared)));
  }
}

template<typename BoundsIterator, typename IndexType>
auto __device__ compute_geometry_id(BoundsIterator bounds_begin, BoundsIterator bounds_end, IndexType idx)
{
    auto it = thrust::prev(
      thrust::upper_bound(thrust::seq, bounds_begin, bounds_end, idx));
    auto const geometry_id = thrust::distance(bounds_begin, it);
    return thrust::make_pair(it, geometry_id);
}

/**
 * @internal
 * @brief The kernel to compute (multi)linestring to (multi)linestring distance
 *
 * Load balanced kernel to compute distances between one pair of segments from the multilinestring
 * and multilinestring.
 *
 * `intersects` is an optional pointer to a boolean range where the `i`th element indicates the
 * `i`th output should be set to 0 and bypass distance computation. This argument is optional, if
 * set to nullopt, no distance computation will be bypassed.
 */
template <typename T, int block_size, class SegmentRange1, class SegmentRange2, class IndexRange, class OutputIt>
__global__ void linestring_distance_load_balanced(SegmentRange1 multilinestrings1,
                                                  SegmentRange2 multilinestrings2,
                                                  IndexRange thread_bounds,
                                                  thrust::optional<uint8_t*> intersects,
                                                  OutputIt distances)
{
  using index_t = typename IndexRange::value_type;

  auto num_segment_pairs = thread_bounds[thread_bounds.size() - 1];
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < num_segment_pairs;
       idx += gridDim.x * blockDim.x) {

    auto [it, geometry_id] = compute_geometry_id(thread_bounds.begin(), thread_bounds.end(), idx);

    auto first_thread_of_this_block = blockDim.x * blockIdx.x;
    auto last_thread_of_block = blockDim.x * (blockIdx.x + 1) - 1;
    auto first_thread_of_next_geometry = thread_bounds[geometry_id + 1];

    bool split_block = first_thread_of_this_block < first_thread_of_next_geometry && first_thread_of_next_geometry <= last_thread_of_block;

    if (intersects.has_value() && intersects.value()[geometry_id]) {
      distances[geometry_id] = 0.0f;
      continue;
    }

    auto const local_idx = idx - *it;
    // Retrieve the number of segments in multilinestrings[geometry_id]
    auto num_segment_this_multilinestring =
      multilinestrings1.multigeometry_count_begin()[geometry_id];
    // The segment id from multilinestring1 this thread is computing (local_id + global_offset)
    auto multilinestring1_segment_id = local_idx % num_segment_this_multilinestring +
                                       multilinestrings1.multigeometry_offset_begin()[geometry_id];

    // The segment id from multilinestring2 this thread is computing (local_id + global_offset)
    auto multilinestring2_segment_id = local_idx / num_segment_this_multilinestring +
                                       multilinestrings2.multigeometry_offset_begin()[geometry_id];

    auto [a, b] = multilinestrings1.begin()[multilinestring1_segment_id];
    auto [c, d] = multilinestrings2.begin()[multilinestring2_segment_id];

    auto partial = sqrt(squared_segment_distance(a, b, c, d));

    if (split_block)
      atomicMin(&distances[geometry_id], partial);
    else
    {
      // block reduce
      typedef cub::BlockReduce<T, block_size> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      auto aggregate = BlockReduce(temp_storage).Reduce(partial, cub::Min());

      // atmomic with leading thread
      if (cooperative_groups::this_thread_block().thread_rank() == 0)
        atomicMin(&distances[geometry_id], aggregate);
    }
  }
}

}  // namespace detail
}  // namespace cuspatial
