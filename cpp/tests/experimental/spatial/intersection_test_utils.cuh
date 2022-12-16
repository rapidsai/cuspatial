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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/experimental/geometry/segment.cuh>

#include <thrust/tuple.h>

namespace cuspatial {
namespace test {

// Custom order for two segments
template <typename T>
bool CUSPATIAL_HOST_DEVICE operator<(segment<T> lhs, segment<T> rhs)
{
  return lhs.v1 < rhs.v1 || (lhs.v1 == rhs.v1 && lhs.v2 < rhs.v2);
}

/**
 * @brief Functor for segmented sorting a geometry array
 *
 * Using a label array and a geometry array as keys, this functor defines that
 * all keys with smaller labels should precede keys with larger labels; and that
 * the order with the same label should be determined by the natural order of the
 * geometries.
 *
 * Example:
 * Labels: {0, 0, 0, 1}
 * Points: {(0, 0), (5, 5), (1, 1), (3, 3)}
 * Result: {(0, 0), (1, 1), (5, 5), (3, 3)}
 */
template <typename KeyType, typename GeomType>
struct order_key_value_pairs {
  using key_value_t = thrust::tuple<KeyType, GeomType>;

  bool CUSPATIAL_HOST_DEVICE operator()(key_value_t lhs, key_value_t rhs)
  {
    return thrust::get<0>(lhs) < thrust::get<0>(rhs) ||
           (thrust::get<0>(lhs) == thrust::get<0>(rhs) &&
            thrust::get<1>(lhs) < thrust::get<1>(rhs));
  }
};

}  // namespace test
}  // namespace cuspatial
