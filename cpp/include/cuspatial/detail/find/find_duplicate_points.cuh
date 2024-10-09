/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cuspatial/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/uninitialized_fill.h>

#include <ranger/ranger.hpp>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief Kernel to compute duplicate points in each multipoint. Naive N^2 algorithm.
 */
template <typename MultiPointRange, typename OutputIt>
CUSPATIAL_KERNEL void find_duplicate_points_kernel_simple(MultiPointRange multipoints,
                                                          OutputIt duplicate_flags)
{
  for (auto idx : ranger::grid_stride_range(multipoints.size())) {
    auto multipoint    = multipoints[idx];
    auto global_offset = multipoints.offsets_begin()[idx];

    // Zero-initialize duplicate_flags for all points in the current space
    for (auto i = 0; i < multipoint.size(); ++i) {
      duplicate_flags[i + global_offset] = 0;
    }

    // Loop over the point range to find duplicates, skipping if the point is already marked as
    // duplicate.
    for (auto i = 0; i < multipoint.size() && duplicate_flags[i + global_offset] != 1; ++i) {
      for (auto j = i + 1; j < multipoint.size(); ++j) {
        if (multipoint[i] == multipoint[j]) duplicate_flags[j + global_offset] = 1;
      }
    }
  }
}

/**
 * @internal
 * @brief For each multipoint, find the duplicate points.
 *
 * If a point has duplicates, all but one flags for the duplicates will be set to 1.
 * There is no guarantee which of the duplicates will not be set.
 */
template <typename MultiPointRange, typename OutputIt>
void find_duplicate_points(MultiPointRange multipoints,
                           OutputIt duplicate_flags,
                           rmm::cuda_stream_view stream)
{
  if (multipoints.size() == 0) return;

  auto [threads_per_block, num_blocks] = grid_1d(multipoints.size());
  find_duplicate_points_kernel_simple<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multipoints, duplicate_flags);

  CUSPATIAL_CHECK_CUDA(stream.value());
}

}  // namespace detail
}  // namespace cuspatial
