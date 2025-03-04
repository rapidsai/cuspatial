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

#include <cuspatial/cuda_utils.hpp>
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
namespace detail {

}  // namespace detail
template <class MultiPointRange, class MultiLinestringRange, class OutputIt>
OutputIt pairwise_point_linestring_distance(MultiPointRange multipoints,
                                            MultiLinestringRange multilinestrings,
                                            OutputIt distances_first,
                                            rmm::cuda_stream_view stream)
{
  using T = typename MultiPointRange::element_t;

  static_assert(is_same_floating_point<T, typename MultiLinestringRange::element_t>(),
                "Inputs must have same floating point value type.");

  static_assert(
    is_same<vec_2d<T>, typename MultiPointRange::point_t, typename MultiLinestringRange::point_t>(),
    "Inputs must be cuspatial::vec_2d");

  CUSPATIAL_EXPECTS(multilinestrings.size() == multipoints.size(),
                    "Input must have the same number of rows.");

  if (multilinestrings.size() == 0) { return distances_first; }

  thrust::fill_n(rmm::exec_policy(stream),
                 distances_first,
                 multilinestrings.size(),
                 std::numeric_limits<T>::max());

  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings.num_points());

  detail::point_linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multipoints, multilinestrings, cuda::std::nullopt, distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return distances_first + multilinestrings.size();
}

}  // namespace cuspatial
