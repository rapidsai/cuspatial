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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/tuple.h>

namespace cuspatial {
namespace detail {

template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename OutputIt1,
          typename OutputIt2>
__global__ void count_intersection_and_overlaps_simple(MultiLinestringRange1 multilinestrings1,
                                                       MultiLinestringRange2 multilinestrings2,
                                                       OutputIt1 point_count_it,
                                                       OutputIt2 segment_count_it)
{
  using T = typename MultiLinestringRange1::element_t;
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);
    auto [a, b]             = multilinestrings1.segment(idx);

    for (auto const& linestring2 : multilinestrings2[geometry_idx]) {
      for (auto [c, d] : linestring2) {
        auto [point_opt, segment_opt] = segment_intersection(segment<T>{a, b}, segment<T>{c, d});
        if (point_opt.has_value()) {
          auto r = make_atomic_ref<cuda::thread_scope_device>(point_count_it[geometry_idx]);
          r.fetch_add(1, cuda::memory_order_relaxed);
        } else if (segment_opt.has_value()) {
          auto r = make_atomic_ref<cuda::thread_scope_device>(segment_count_it[geometry_idx]);
          r.fetch_add(1, cuda::memory_order_relaxed);
        }
      }
    }
  }
}

/**
 * @internal
 * @brief Count the upper bound of intersecting linestrings between a pair of multilinestring range
 *
 * @tparam MultiLinestringRange1 Type of first multilinestring range
 * @tparam MultiLinestringRange2 Type of second multilinestring range
 * @tparam OutputIt1 Type of intersecting point count iterator
 * @tparam OutputIt2 Type of overlapping segment count iterator
 * @param multilinestrings1 The first multilinestring range
 * @param multilinestrings2 The second multilinestring range
 * @param points_count_it Integral iterator to the number of intersecting points
 * @param segments_count_it Integral iterator to the number of overlapping segments
 * @param stream The CUDA stream for device memory operations
 */
template <typename MultiLinestringRange1,
          typename MultiLinestringRange2,
          typename OutputIt1,
          typename OutputIt2>
void pairwise_linestring_intersection_upper_bound_count(MultiLinestringRange1 multilinestrings1,
                                                        MultiLinestringRange2 multilinestrings2,
                                                        OutputIt1 points_count_it,
                                                        OutputIt2 segments_count_it,
                                                        rmm::cuda_stream_view stream)
{
  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings1.size());
  detail::
    count_intersection_and_overlaps_simple<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      multilinestrings1, multilinestrings2, points_count_it, segments_count_it);
}

}  // namespace detail
}  // namespace cuspatial
