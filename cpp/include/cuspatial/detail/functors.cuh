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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/traits.hpp>

#include <cuda/std/utility>
#include <thrust/binary_search.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Given iterator a pair of offsets, return the number of elements between the offsets.
 *
 * Used to create iterator to geometry counts, such as `multi*_point_count_begin`,
 * `multi*_segment_count_begin`.
 *
 * Example:
 * pair of offsets: (0, 3), (3, 5), (5, 8)
 * number of elements between offsets: 3, 2, 3
 *
 * @tparam OffsetPairIterator Must be iterator type to cuda::std::pair of indices.
 * @param p Iterator of cuda::std::pair of indices.
 */
struct offset_pair_to_count_functor {
  template <typename OffsetPairIterator>
  CUSPATIAL_HOST_DEVICE auto operator()(OffsetPairIterator p)
  {
    return thrust::get<1>(p) - thrust::get<0>(p);
  }
};

}  // namespace detail
}  // namespace cuspatial
