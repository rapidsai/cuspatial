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

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

namespace cuspatial {
namespace detail {

inline __device__ std::pair<uint32_t, uint32_t> get_quad_and_local_point_indices(
  uint32_t const global_index, uint32_t const* point_offsets, uint32_t const* point_offsets_end)
{
  // Calculate the position in "point_offsets" that `global_index` falls between.
  // This position is the index of the poly/quad pair for this `global_index`.
  //
  // Dereferencing `local_point_offset` yields the zero-based first point position of this
  // quadrant. Adding this zero-based position to the quadrant's first point position in the
  // quadtree yields the "global" position in the `point_indices` map.
  auto const local_point_offset =
    thrust::upper_bound(thrust::seq, point_offsets, point_offsets_end, global_index) - 1;
  return std::make_pair(
    // quad_poly_index
    thrust::distance(point_offsets, local_point_offset),
    // local_point_index
    global_index - *local_point_offset);
}

}  // namespace detail
}  // namespace cuspatial
