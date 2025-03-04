/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cuspatial/detail/kernel/pairwise_distance.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/optional>
#include <thrust/fill.h>

#include <limits>
#include <type_traits>

namespace cuspatial {

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

  if (multilinestrings1.size() == 0) return distances_first;

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + multilinestrings1.size(),
               std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 256;
  std::size_t const num_blocks =
    (multilinestrings1.num_points() + threads_per_block - 1) / threads_per_block;

  detail::linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multilinestrings1, multilinestrings2, cuda::std::nullopt, distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
  return distances_first + multilinestrings1.size();
}

}  // namespace cuspatial
