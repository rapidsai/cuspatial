/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
inline __device__ bool is_point_in_polygon(T const x,
                                           T const y,
                                           cudf::size_type const poly_idx,
                                           cudf::column_device_view const& poly_offsets,
                                           cudf::column_device_view const& ring_offsets,
                                           cudf::column_device_view const& poly_points_x,
                                           cudf::column_device_view const& poly_points_y)
{
  bool in_polygon     = false;
  uint32_t poly_begin = poly_offsets.element<uint32_t>(poly_idx);
  uint32_t poly_end   = poly_idx < poly_offsets.size() - 1
                          ? poly_offsets.element<uint32_t>(poly_idx + 1)
                          : ring_offsets.size();

  for (uint32_t ring_idx = poly_begin; ring_idx < poly_end; ring_idx++)  // for each ring
  {
    auto ring_begin = ring_offsets.element<uint32_t>(ring_idx);
    auto ring_end   = ring_idx < ring_offsets.size() - 1
                        ? ring_offsets.element<uint32_t>(ring_idx + 1)
                        : poly_points_x.size();
    auto ring_len   = ring_end - ring_begin;
    for (auto point_idx = 0; point_idx < ring_len; ++point_idx) {
      T x0                 = poly_points_x.element<T>(ring_begin + ((point_idx + 0) % ring_len));
      T y0                 = poly_points_y.element<T>(ring_begin + ((point_idx + 0) % ring_len));
      T x1                 = poly_points_x.element<T>(ring_begin + ((point_idx + 1) % ring_len));
      T y1                 = poly_points_y.element<T>(ring_begin + ((point_idx + 1) % ring_len));
      bool y_between_ay_by = y0 <= y && y < y1;  // is y in range [ay, by) when ay < by
      bool y_between_by_ay = y1 <= y && y < y0;  // is y in range [by, ay) when by < ay
      bool y_in_bounds     = y_between_ay_by || y_between_by_ay;  // is y in range [by, ay]
      T run                = x1 - x0;
      T rise               = y1 - y0;
      T rise_to_point      = y - y0;

      if (y_in_bounds && x < (run / rise) * rise_to_point + x0) { in_polygon = not in_polygon; }
    }
  }

  return in_polygon;
}

}  // namespace detail
}  // namespace cuspatial
