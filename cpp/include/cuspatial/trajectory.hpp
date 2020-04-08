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
#include <rmm/mr/device/default_memory_resource.hpp>

namespace cuspatial {
namespace experimental {

/**
 * @brief Derive trajectories from sorted object ids.
 *
 * Groups the input object ids to determine unique trajectories. Returns a
 * table with the trajectory ids, the number of objects in each trajectory,
 * and the offset position of the first object for each trajectory in the
 * input object ids column.
 *
 * @param[in] id column of object (e.g., vehicle) ids
 * @param[in] mr The optional resource to use for all allocations
 *
 * @return a sorted table with the following three int32 columns:
 *   * trajectory id - the unique ids from the input object ids column
 *   * trajectory length - the number of objects in the derived trajectories
 *   * trajectory offset - the cumulative sum of start positions for each group
 */
std::unique_ptr<cudf::experimental::table> derive_trajectories(
    cudf::column_view const& id,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

/**
 * @brief Compute the distance and speed of trajectories
 *
 * Trajectories are derived from coordinate data using `derive_trajectories`.
 *
 * @param[in] x coordinates (km) (sorted by id, timestamp)
 * @param[in] y coordinates (km) (sorted by id, timestamp)
 * @param[in] timestamp column (sorted by id, timestamp)
 * @param[in] length the number of points column (sorted by id, timestamp)
 * @param[in] offset position of each trajectory's first object, used to index
 * timestamp/x/y columns (sorted by id, timestamp)
 * @param[in] mr The optional resource to use for all allocations
 *
 * @return a sorted cudf table of distances (meters) and speeds (meters/second)
 */
std::unique_ptr<cudf::experimental::table> compute_velocities(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& timestamp, cudf::column_view const& length,
    cudf::column_view const& offset,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cuspatial
