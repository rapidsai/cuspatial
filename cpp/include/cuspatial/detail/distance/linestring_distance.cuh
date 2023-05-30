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

#include <cuspatial/detail/algorithm/linestring_distance.cuh>
#include <cuspatial/detail/distance/distance_utils.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/range/range.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/optional.h>

#include <limits>
#include <type_traits>

namespace cuspatial {

template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
OutputIt pairwise_linestring_distance(MultiLinestringRange1 lhs,
                                      MultiLinestringRange2 rhs,
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

  CUSPATIAL_EXPECTS(lhs.size() == rhs.size(), "Inputs must have the same number of rows.");

  if (lhs.size() == 0) return distances_first;

  // Make views to the segments in the multilinestring
  auto lhs_segments       = lhs._segments(stream);
  auto lhs_segments_range = lhs_segments.view();

  // Make views to the segments in the multilinestring
  auto rhs_segments       = rhs._segments(stream);
  auto rhs_segments_range = rhs_segments.view();

  auto thread_bounds =
    detail::compute_segment_thread_bounds(lhs_segments_range.multigeometry_count_begin(),
                                          lhs_segments_range.multigeometry_count_end(),
                                          rhs_segments_range.multigeometry_count_begin(),
                                          stream);
  // Initialize the output range
  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + lhs.size(),
               std::numeric_limits<T>::max());

  std::size_t constexpr threads_per_block = 256;
  // std::size_t num_threads                 = thread_bounds.element(thread_bounds.size() - 1, stream);
  std::size_t num_threads = lhs.num_points() * rhs.num_points();
  std::size_t const num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

  detail::linestring_distance_load_balanced<T, threads_per_block>
    <<<num_blocks, threads_per_block, 0, stream.value()>>>(
      lhs_segments_range,
      rhs_segments_range,
      range(thread_bounds.begin(), thread_bounds.end()),
      thrust::nullopt,
      distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
  return distances_first + lhs.size();
}

}  // namespace cuspatial
