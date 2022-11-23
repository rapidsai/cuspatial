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
void __global__ simple_combine_segments_kernel(OffsetRange offsets,
                                               SegmentRange segments,
                                               OutputIt stencils)
{
  for (auto pair_idx = threadIdx.x + blockIdx.x * blockDim.x; pair_idx < offsets.size() - 1;
       pair_idx += gridDim.x * blockDim.x) {
    for (auto i = offsets[pair_idx]; i < offsets[pair_idx + 1] && stencils[i] != 1; i++) {
      for (auto j = i + 1; j < offsets[pair_idx + 1]; j++) {
        auto res = maybe_merge_segments(segments[i], segments[j]);
        if (res.has_value()) {
          // segments[i] can be merged from segments[j]
          segments[i] = res.value();
          stencils[j] = 1;
        }
      }
    }
  }
}

/**
 * @internal
 * @brief For each pair of mergeable segment, overwrites the first segment with merged result,
 * set the stencil for the second segment in the output stencil.
 */
template <typename OffsetRange, typename SegmentRange, typename OutputIt>
void combine_segments(OffsetRange offsets,
                      SegmentRange segments,
                      OutputIt stencils_output,
                      rmm::cuda_stream_view stream)
{
  auto [threads_per_block, num_blocks] = grid_1d(segments.size());
  thrust::uninitialized_fill_n(rmm::exec_policy(stream), stencils_output, segments.size(), 0);
  simple_combine_segments_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    offsets, segments, stencils_output);
}

}  // namespace detail
}  // namespace cuspatial
