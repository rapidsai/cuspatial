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

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/memory.h>

#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @brief Kernel to compute the distance between pairs of point and linestring.
 *
 * The kernel is launched on one linestring point per thread. Each thread iterates on all points in
 * the multipoint operand and use atomics to aggregate the shortest distance.
 */
template <class MultiPointRange, class MultiLinestringRange, class OutputIterator>
void __global__ pairwise_point_linestring_distance_kernel(MultiPointRange multipoints,
                                                          MultiLinestringRange multilinestrings,
                                                          OutputIterator distances)
{
  using T = typename MultiPointRange::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings.num_points();
       idx += gridDim.x * blockDim.x) {
    // Search from the part offsets array to determine the part idx of current linestring point
    auto part_idx = multilinestrings.part_idx_from_point_idx(idx);
    // Pointer to the last point in the linestring, skip iteration.
    // Note that the last point for the last linestring is guarded by the grid-stride loop.
    if (!multilinestrings.is_valid_segment_id(idx, part_idx)) continue;

    // Search from the linestring geometry offsets array to determine the geometry idx of
    // current linestring point
    auto geometry_idx = multilinestrings.geometry_idx_from_part_idx(part_idx);

    // Reduce the minimum distance between different parts of the multi-point.
    auto [a, b]            = multilinestrings.segment(idx);
    T min_distance_squared = std::numeric_limits<T>::max();

    for (vec_2d<T> const& c : multipoints[geometry_idx]) {
      // TODO: reduce redundant computation only related to `a`, `b` in this helper.
      auto const distance_squared = point_to_segment_distance_squared(c, a, b);
      min_distance_squared        = min(distance_squared, min_distance_squared);
    }
    atomicMin(&distances[geometry_idx], static_cast<T>(sqrt(min_distance_squared)));
  }
}

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

  std::size_t constexpr threads_per_block = 256;
  std::size_t const num_blocks =
    (multilinestrings.size() + threads_per_block - 1) / threads_per_block;

  detail::
    pairwise_point_linestring_distance_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      multipoints, multilinestrings, distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return distances_first + multilinestrings.size();
}

}  // namespace cuspatial
