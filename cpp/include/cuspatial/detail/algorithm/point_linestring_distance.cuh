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

#pragma once

#include <cuspatial/geometry/linestring_ref.cuh>
#include <cuspatial/geometry/segment.cuh>
#include <cuspatial/geometry/vec_2d.hpp>

namespace cuspatial {
namespace detail {

template <typename T>
__device__ T proj2(segment<T> const& s, vec_2d<T> const& v)
{
  return dot(v - s.v1, s.v2 - s.v1);
}

template <typename T, typename LinestringRef>
inline __device__ T point_linestring_distance(vec_2d<T> const& point,
                                              LinestringRef const& linestring)
{
  T distance_squared = std::numeric_limits<T>::max();

  for (auto const& s : linestring) {
    auto v1p         = point - s.v1;
    auto v2p         = point - s.v2;
    auto d0          = dot(v1p, v1p);
    auto d1          = dot(v2p, v2p);
    auto d2          = s.length2();
    auto d3          = proj2(s, point);
    auto const r     = d3 * d3 / d2;
    auto const d     = (d3 <= 0 || r >= d2) ? min(d0, d1) : d0 - r;
    distance_squared = min(distance_squared, d);
  }

  return sqrt(distance_squared);
}

}  // namespace detail
}  // namespace cuspatial
