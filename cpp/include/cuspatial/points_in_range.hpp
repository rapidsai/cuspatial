/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial_relationship
 */

/**
 * @brief Find all points (x,y) that fall within a rectangular query range.
 *
 * A point (x, y) is in the range if `x > range_min_x && x < range_min_y && y > range_min_y && y
 * < range_max_y`.
 *
 * Swaps `range_min_x` and `range_max_x` if `range_min_x > range_max_x`.
 * Swaps `range_min_y` and `range_max_y` if `range_min_y > range_max_y`.
 *
 * The range coordinates and the (x, y) points to be tested are assumed to be defined in the same
 * coordinate system.
 *
 * @param[in] range_min_x lower x-coordinate of the query range
 * @param[in] range_max_x upper x-coordinate of the query range
 * @param[in] range_min_y lower y-coordinate of the query range
 * @param[in] range_max_y upper y-coordinate of the query range
 * @param[in] x            x-coordinates of points to be queried
 * @param[in] y            y-coordinates of points to be queried
 * @param[in] mr           Optional `device_memory_resource` to use for allocating the output table
 *
 * @returns A table with two columns of the same type as the input columns. Columns 0, 1 are the
 * (x, y) coordinates of the points in the input that fall within the query range.
 */
std::unique_ptr<cudf::table> points_in_range(
  double range_min_x,
  double range_max_x,
  double range_min_y,
  double range_max_y,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
