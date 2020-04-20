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

#include <cuspatial/error.hpp>
#include <cuspatial/trajectory.hpp>

namespace cuspatial {
namespace experimental {
namespace detail {

/**
 * @brief Derive trajectories from points, timestamps, and object ids.
 *
 * Groups the input object ids to determine unique trajectories. Returns a
 * table with the trajectory ids, the number of objects in each trajectory,
 * and the offset position of the first object for each trajectory in the
 * input object ids column.
 *
 * @param[in] x coordinates (km) (sorted by id, timestamp)
 * @param[in] y coordinates (km) (sorted by id, timestamp)
 * @param[in] object_id column of object (e.g., vehicle) ids
 * @param[in] timestamp column (sorted by id, timestamp)
 * @param[in] mr The optional resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to schedule allocations
 *
 * @return an `std::pair<table, column>`:
 *  1. table of (object_id, timestamp, x, y) sorted by (object_id, timestamp)
 *  2. int32 column of end positions for each trajectory's last object
 */
std::pair<std::unique_ptr<cudf::experimental::table>,
          std::unique_ptr<cudf::column>>
derive_trajectories(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& object_id, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Compute the distance and speed of trajectories.
 *
 * Trajectories are derived from coordinate data using
 * `derive_trajectories`.
 *
 * @param[in] x coordinates (km) (sorted by id, timestamp)
 * @param[in] y coordinates (km) (sorted by id, timestamp)
 * @param[in] timestamp column (sorted by id, timestamp)
 * @param[in] end position for each trajectory's last object, used to index
 * timestamp/x/y columns (sorted by id, timestamp)
 * @param[in] mr The optional resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to schedule allocations
 *
 * @return a sorted cudf table of distances (meters) and speeds (meters/second)
 */
std::unique_ptr<cudf::experimental::table> trajectory_distance_and_speed(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& object_id, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Compute the spatial bounding boxes of trajectories.
 *
 * Trajectories are derived from coordinate data using
 * `derive_trajectories`.
 *
 * @param[in] x coordinates (km) (sorted by id, timestamp)
 * @param[in] y coordinates (km) (sorted by id, timestamp)
 * @param[in] end position for each trajectory's last object, used to index
 * timestamp/x/y columns (sorted by id, timestamp)
 * @param[in] mr The optional resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to schedule allocations
 *
 * @return a cudf table of bounding boxes with four columns:
 *   * x1 - the lower-left x-coordinate of each bounding box
 *   * y1 - the lower-left y-coordinate of each bounding box
 *   * x2 - the upper-right x-coordinate of each bounding box
 *   * y2 - the upper-right y-coordinate of each bounding box
 */
std::unique_ptr<cudf::experimental::table> trajectory_bounding_boxes(
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& object_id,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cuspatial
