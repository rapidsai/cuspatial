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
#include <cuspatial/detail/utility/traits.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/iterator_collections.cuh>
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

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIterator>
void __global__ pairwise_point_linestring_distance(
  iterator_collections::multipoint_array<OffsetIteratorA, Cart2dItA> multipoints,
  iterator_collections::multilinestring_array<OffsetIteratorB, OffsetIteratorC, Cart2dItB>
    multilinestrings,
  OutputIterator distances)
{
  using T = iterator_vec_base_type<Cart2dItA>;

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

    for (vec_2d<T> const& c : multipoints.element(geometry_idx)) {
      // TODO: reduce redundant computation only related to `a`, `b` in this helper.
      auto const distance_squared = point_to_segment_distance_squared(c, a, b);
      min_distance_squared        = std::min(distance_squared, min_distance_squared);
    }
    atomicMin(&thrust::raw_reference_cast(*(distances + geometry_idx)),
              static_cast<T>(std::sqrt(min_distance_squared)));
  }
}

}  // namespace detail
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_distance(
  iterator_collections::multipoint_array<OffsetIteratorA, Cart2dItA> multipoints,
  iterator_collections::multilinestring_array<OffsetIteratorB, OffsetIteratorC, Cart2dItB>
    multilinestrings,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default)
{
  using T = detail::iterator_vec_base_type<Cart2dItA>;

  static_assert(detail::is_same_floating_point<T, detail::iterator_vec_base_type<Cart2dItB>>(),
                "Inputs must have same floating point value type.");

  static_assert(detail::is_same<vec_2d<T>,
                                detail::iterator_value_type<Cart2dItA>,
                                detail::iterator_value_type<Cart2dItB>>(),
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

  detail::pairwise_point_linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multipoints, multilinestrings, distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return distances_first + multilinestrings.size();
}

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_distance(OffsetIteratorA point_geometry_offset_first,
                                            OffsetIteratorA point_geometry_offset_last,
                                            Cart2dItA points_first,
                                            Cart2dItA points_last,
                                            OffsetIteratorB linestring_geometry_offset_first,
                                            OffsetIteratorC linestring_part_offsets_first,
                                            OffsetIteratorC linestring_part_offsets_last,
                                            Cart2dItB linestring_points_first,
                                            Cart2dItB linestring_points_last,
                                            OutputIt distances_first,
                                            rmm::cuda_stream_view stream)
{
  auto d = thrust::distance(point_geometry_offset_first, point_geometry_offset_first);
  return pairwise_point_linestring_distance(
    iterator_collections::multipoint_array(
      point_geometry_offset_first, point_geometry_offset_last, points_first, points_last),
    iterator_collections::multilinestring_array(linestring_geometry_offset_first,
                                                linestring_geometry_offset_first + d,
                                                linestring_part_offsets_first,
                                                linestring_part_offsets_last,
                                                linestring_points_first,
                                                linestring_points_last),
    distances_first,
    stream

  );
}

}  // namespace cuspatial
