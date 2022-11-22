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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/uninitialized_fill.h>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief Kernel to compute duplicate points in each multipoint. Naive N^2 algorithm.
 */
template <typename MultiPointRange, typename OutputIt>
void __global__ combine_duplicate_points_kernel_simple(MultiPointRange multipoints,
                                                       OutputIt stencils)
{
  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multipoints.num_points();
       idx += gridDim.x * blockDim.x) {
    auto multipoint    = multipoints[idx];
    auto global_offset = multipoints.offsets_begin()[idx];
    for (auto i = 0; i < multipoint.size() && stencils[i] != 1; ++i)
      for (auto j = i + 1; j < multipoint.size(); ++j) {
        if (multipoint[i] == multipoint[j]) stencils[j + global_offset] = 1;
      }
  }
}

/**
 * @internal
 * @brief For each multipoint, computes duplicate points and stores as stencil.
 */
template <typename MultiPointRange, typename OutputIt>
void combine_duplicate_points(MultiPointRange multipoints,
                              OutputIt output_stencils,
                              rmm::cuda_stream_view stream)
{
  if (multipoints.size() == 0) return;

  thrust::uninitialized_fill_n(
    rmm::exec_policy(stream), output_stencils, multipoints.num_points(), 0);

  auto [threads_per_block, num_blocks] = grid_1d(multipoints.size());
  combine_duplicate_points_kernel_simple<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multipoints, output_stencils);
}

}  // namespace detail
}  // namespace cuspatial
