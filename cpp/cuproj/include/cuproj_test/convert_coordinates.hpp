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

#include <proj.h>

#include <algorithm>
#include <type_traits>

namespace cuproj_test {

// Convert coordinates from a x-y struct to a PJ_COORD struct or vice versa
template <typename InVector, typename OutVector>
void convert_coordinates(InVector const& in, OutVector& out)
{
  using in_coord_type  = typename InVector::value_type;
  using out_coord_type = typename OutVector::value_type;

  static_assert(
    (std::is_same_v<out_coord_type, PJ_COORD> != std::is_same_v<in_coord_type, PJ_COORD>),
    "Invalid coordinate vector conversion");

  if constexpr (std::is_same_v<in_coord_type, PJ_COORD>) {
    using T                       = typename out_coord_type::value_type;
    auto proj_coord_to_coordinate = [](auto const& c) {
      return out_coord_type{static_cast<T>(c.xy.x), static_cast<T>(c.xy.y)};
    };
    std::transform(in.begin(), in.end(), out.begin(), proj_coord_to_coordinate);
  } else if constexpr (std::is_same_v<out_coord_type, PJ_COORD>) {
    auto coordinate_to_proj_coord = [](auto const& c) { return PJ_COORD{c.x, c.y, 0, 0}; };
    std::transform(in.begin(), in.end(), out.begin(), coordinate_to_proj_coord);
  }
}

}  // namespace cuproj_test
