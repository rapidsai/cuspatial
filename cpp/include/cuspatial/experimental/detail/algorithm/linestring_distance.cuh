/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/optional.h>

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
 */
template <class MultiLinestringRange1, class MultiLinestringRange2, class OutputIt>
__global__ void linestring_distance(MultiLinestringRange1 multilinestrings1,
                                    MultiLinestringRange2 multilinestrings2,
                                    thrust::optional<uint8_t*> intersects,
                                    OutputIt distances_first)
{
  using T = typename MultiLinestringRange1::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multilinestrings1.num_points();
       idx += gridDim.x * blockDim.x) {
    auto const part_idx = multilinestrings1.part_idx_from_point_idx(idx);
    if (!multilinestrings1.is_valid_segment_id(idx, part_idx)) continue;
    auto const geometry_idx = multilinestrings1.geometry_idx_from_part_idx(part_idx);

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

}  // namespace detail
}  // namespace cuspatial
