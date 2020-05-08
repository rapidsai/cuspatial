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

#include <cudf/types.hpp>
#include <memory>

namespace cuspatial {

/**
 * @brief Find all points (x,y) that fall within a query window (left, bottom, right, top).
 *
 * A point (x, y) is in the window if `x > left && x < right && y > bottom && y < top`.
 *
 * @param[in] left   x-coordinate of left edge of the query window
 * @param[in] bottom y-coordinate of bottom of the query window
 * @param[in] right  x-coordinate of right edge of the query window
 * @param[in] top    y-coordinate of top of the query window
 * @param[in] x      x-coordinates of points to be queried
 * @param[in] y      y-coordinates of points to be queried
 * @param[in] mr     Optional `device_memory_resource` to use for allocating the output table
 *
 * @returns A table with two columns of the same type as the input columns. Columns 0, 1 are the
 * (x, y) coordinates of the points in the input that fall within the query window.
 */
std::unique_ptr<cudf::experimental::table> points_in_spatial_window(
  double left,
  double bottom,
  double right,
  double top,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
