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

#include <thrust/fill.h>

#include <limits>
#include <type_traits>

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
 * which is then combined with other segment results via an atomic operation to form the globally
 * minimum distance between the linestrings.
 */
template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
__global__ void pairwise_linestring_distance_kernel(MultiLinestringRange1 multilinestrings1,
                                                    MultiLinestringRange2 multilinestrings2,
                                                    OutputIt distances_first)
{
  using T = typename MultiLinestringRange1::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);
    auto [a, b]             = multilinestrings1.segment(idx);
    T min_distance_squared  = std::numeric_limits<T>::max();

    for (auto const& linestring2 : multilinestrings2[geometry_idx]) {
      for (auto [c, d] : linestring2) {
        min_distance_squared = min(min_distance_squared, squared_segment_distance(a, b, c, d));
      }
    }
    atomicMin(&distances_first[geometry_idx], static_cast<T>(sqrt(min_distance_squared)));
  }
}

}  // namespace detail

template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
OutputIt pairwise_linestring_distance(MultiLinestringRange1 multilinestrings1,
                                      MultiLinestringRange2 multilinestrings2,
                                      OutputIt distances_first,
                                      rmm::cuda_stream_view stream)
{
  using T = typename MultiLinestringRange1::element_t;

  static_assert(is_same_floating_point<T, typename MultiLinestringRange2::element_t>(),
                "Inputs and output must have the same floating point value type.");

  static_assert(is_same<vec_2d<T>,
                        typename MultiLinestringRange1::point_t,
                        typename MultiLinestringRange2::point_t>(),
                "All input types must be cuspatial::vec_2d with the same value type");

  CUSPATIAL_EXPECTS(multilinestrings1.size() == multilinestrings2.size(),
                    "Inputs must have the same number of rows.");

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + multilinestrings1.size(),
               std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 256;
  std::size_t const num_blocks =
    (multilinestrings1.num_points() + threads_per_block - 1) / threads_per_block;

  detail::pairwise_linestring_distance_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multilinestrings1, multilinestrings2, distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
  return distances_first + multilinestrings1.size();
}

}  // namespace cuspatial
