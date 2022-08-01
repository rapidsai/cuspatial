/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @brief Find all points (x,y) that fall within a rectangular query window.
 *
 * @ingroup spatial_relationship
 *
 * A point (x, y) is in the window if `x > window_min_x && x < window_min_y && y > window_min_y && y
 * < window_max_y`.
 *
 * Swaps `window_min_x` and `window_max_x` if `window_min_x > window_max_x`.
 * Swaps `window_min_y` and `window_max_y` if `window_min_y > window_max_y`.
 *
 * The window coordinates and the (x, y) points to be tested are assumed to be defined in the same
 * coordinate system.
 *
 * @param[in] window_min_x lower x-coordinate of the query window
 * @param[in] window_max_x upper x-coordinate of the query window
 * @param[in] window_min_y lower y-coordinate of the query window
 * @param[in] window_max_y upper y-coordinate of the query window
 * @param[in] x            x-coordinates of points to be queried
 * @param[in] y            y-coordinates of points to be queried
 * @param[in] mr           Optional `device_memory_resource` to use for allocating the output table
 *
 * @returns A table with two columns of the same type as the input columns. Columns 0, 1 are the
 * (x, y) coordinates of the points in the input that fall within the query window.
 */
std::unique_ptr<cudf::table> points_in_spatial_window(
  double window_min_x,
  double window_max_x,
  double window_min_y,
  double window_max_y,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
