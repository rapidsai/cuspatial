/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>

#include <ranger/ranger.hpp>

#include <limits>

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
 * which is then combined with other segment results via an atomic operation to form the global
 * minimum distance between the linestrings.
 *
 * `intersects` is an optional pointer to a boolean range where the `i`th element indicates the
 * `i`th output should be set to 0 and bypass distance computation. This argument is optional, if
 * set to nullopt, no distance computation will be bypassed.
 *
 * @note This kernel does not compute pairs that contains empty geometry.
 */
template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
CUSPATIAL_KERNEL void linestring_distance(MultiLinestringRange1 multilinestrings1,
                                          MultiLinestringRange2 multilinestrings2,
                                          cuda::std::optional<uint8_t*> intersects,
                                          OutputIt distances_first)
{
  using T = typename MultiLinestringRange1::element_t;

  for (auto idx : ranger::grid_stride_range(multilinestrings1.num_points())) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);

    if (multilinestrings1[geometry_idx].is_empty() || multilinestrings2[geometry_idx].is_empty()) {
      continue;
    }

    if (intersects.has_value() && intersects.value()[geometry_idx]) {
      distances_first[geometry_idx] = 0;
      continue;
    }
    auto [a, b]            = multilinestrings1.segment(idx);
    T min_distance_squared = std::numeric_limits<T>::max();

    for (auto const& linestring2 : multilinestrings2[geometry_idx]) {
      for (auto [c, d] : linestring2) {
        min_distance_squared = min(min_distance_squared, squared_segment_distance(a, b, c, d));
      }
    }
    atomicMin(&distances_first[geometry_idx], static_cast<T>(sqrt(min_distance_squared)));
  }
}

/**
 * @brief Kernel to compute the distance between pairs of point and linestring.
 *
 * The kernel is launched on one linestring point per thread. Each thread iterates on all points in
 * the multipoint operand and use atomics to aggregate the shortest distance.
 *
 * `intersects` is an optional pointer to a boolean range where the `i`th element indicates the
 * `i`th output should be set to 0 and bypass distance computation. This argument is optional, if
 * set to nullopt, no distance computation will be bypassed.
 */
template <class MultiPointRange, class MultiLinestringRange, class OutputIterator>
CUSPATIAL_KERNEL void point_linestring_distance(MultiPointRange multipoints,
                                                MultiLinestringRange multilinestrings,
                                                cuda::std::optional<uint8_t*> intersects,
                                                OutputIterator distances)
{
  using T = typename MultiPointRange::element_t;

  for (auto idx : ranger::grid_stride_range(multilinestrings.num_points())) {
    // Search from the part offsets array to determine the part idx of current linestring point
    auto part_idx = multilinestrings.part_idx_from_point_idx(idx);
    // Pointer to the last point in the linestring, skip iteration.
    // Note that the last point for the last linestring is guarded by the grid-stride loop.
    if (!multilinestrings.is_valid_segment_id(idx, part_idx)) continue;

    // Search from the linestring geometry offsets array to determine the geometry idx of
    // current linestring point
    auto geometry_idx = multilinestrings.geometry_idx_from_part_idx(part_idx);

    if (intersects.has_value() && intersects.value()[geometry_idx]) {
      distances[geometry_idx] = 0;
      continue;
    }

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
}  // namespace cuspatial
