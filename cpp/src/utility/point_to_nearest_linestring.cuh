/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

namespace cuspatial {
namespace detail {

template <typename T>
inline __device__ T
point_to_linestring_distance(T const px,
                             T const py,
                             cudf::size_type const ring_idx,
                             cudf::column_device_view const& ring_offsets,
                             cudf::column_device_view const& linestring_points_x,
                             cudf::column_device_view const& linestring_points_y)
{
  T distance_squared = std::numeric_limits<T>::max();
  auto ring_begin    = ring_offsets.element<uint32_t>(ring_idx);
  auto ring_end = ring_idx < ring_offsets.size() - 1 ? ring_offsets.element<uint32_t>(ring_idx + 1)
                                                     : linestring_points_x.size();
  auto ring_len = ring_end - ring_begin;
  for (auto point_idx = 0u; point_idx < ring_len; ++point_idx) {
    auto const i0    = ring_begin + ((point_idx + 0) % ring_len);
    auto const i1    = ring_begin + ((point_idx + 1) % ring_len);
    auto const x0    = linestring_points_x.element<T>(i0);
    auto const y0    = linestring_points_y.element<T>(i0);
    auto const x1    = linestring_points_x.element<T>(i1);
    auto const y1    = linestring_points_y.element<T>(i1);
    auto const dx0   = px - x0;
    auto const dy0   = py - y0;
    auto const dx1   = px - x1;
    auto const dy1   = py - y1;
    auto const dx2   = x1 - x0;
    auto const dy2   = y1 - y0;
    auto const d0    = dx0 * dx0 + dy0 * dy0;
    auto const d1    = dx1 * dx1 + dy1 * dy1;
    auto const d2    = dx2 * dx2 + dy2 * dy2;
    auto const d3    = dx2 * dx0 + dy2 * dy0;
    auto const r     = d3 * d3 / d2;
    auto const d     = d3 <= 0 || r >= d2 ? min(d0, d1) : d0 - r;
    distance_squared = min(distance_squared, d);
  }

  return sqrt(distance_squared);
}

}  // namespace detail
}  // namespace cuspatial
