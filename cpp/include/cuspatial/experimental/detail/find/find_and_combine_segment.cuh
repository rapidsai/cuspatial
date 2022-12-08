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
#include <cuspatial/detail/utility/linestring.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/uninitialized_fill.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Kernel to merge segments, naive n^2 algorithm.
 */
template <typename OffsetRange, typename SegmentRange, typename OutputIt>
void __global__ simple_find_and_combine_segments_kernel(OffsetRange offsets,
                                                        SegmentRange segments,
                                                        OutputIt merged_flag)
{
  for (auto pair_idx = threadIdx.x + blockIdx.x * blockDim.x; pair_idx < offsets.size() - 1;
       pair_idx += gridDim.x * blockDim.x) {
    // Zero-initialize flags for all segments in current space.
    for (auto i = offsets[pair_idx]; i < offsets[pair_idx + 1]; i++) {
      merged_flag[i] = 0;
    }

    for (auto i = offsets[pair_idx]; i < offsets[pair_idx + 1] && merged_flag[i] != 1; i++) {
      for (auto j = i + 1; j < offsets[pair_idx + 1]; j++) {
        auto res = maybe_merge_segments(segments[i], segments[j]);
        if (res.has_value()) {
          // segments[i] can be merged from segments[j]
          segments[i]    = res.value();
          merged_flag[j] = 1;
        }
      }
    }
  }
}

/**
 * @internal
 * @brief For each pair of mergeable segment, overwrites the first segment with merged result,
 * sets the flag for the second segment as 1.
 */
template <typename OffsetRange, typename SegmentRange, typename OutputIt>
void find_and_combine_segment(OffsetRange offsets,
                              SegmentRange segments,
                              OutputIt merged_flag,
                              rmm::cuda_stream_view stream)
{
  auto num_spaces = offsets.size() - 1;
  if (num_spaces == 0) return;

  auto [threads_per_block, num_blocks] = grid_1d(num_spaces);
  simple_find_and_combine_segments_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    offsets, segments, merged_flag);
}

}  // namespace detail
}  // namespace cuspatial
