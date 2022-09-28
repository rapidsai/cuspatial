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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup trajectory_api
 * @{
 */

/**
 * @brief Derive trajectories from object ids, points, and timestamps.
 *
 * Groups the input object ids to determine unique trajectories. Returns a
 * table with the trajectory ids, the number of objects in each trajectory,
 * and the offset position of the first object for each trajectory in the
 * input object ids column.
 *
 * @param object_id column of object (e.g., vehicle) ids
 * @param x coordinates (in kilometers)
 * @param y coordinates (in kilometers)
 * @param timestamp column of timestamps in any resolution
 * @param mr The optional resource to use for output device memory allocations
 *
 * @throw cuspatial::logic_error If object_id isn't cudf::type_id::INT32
 * @throw cuspatial::logic_error If x and y are different types
 * @throw cuspatial::logic_error If timestamp isn't a cudf::TIMESTAMP type
 * @throw cuspatial::logic_error If object_id, x, y, or timestamp contain nulls
 * @throw cuspatial::logic_error If object_id, x, y, and timestamp are different
 * sizes
 *
 * @return an `std::pair<table, column>`:
 *  1. table of (object_id, x, y, timestamp) sorted by (object_id, timestamp)
 *  2. int32 column of start positions for each trajectory's first object
 */
std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> derive_trajectories(
  cudf::column_view const& object_id,
  cudf::column_view const& x,
  cudf::column_view const& y,
  cudf::column_view const& timestamp,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute the distance and speed of objects in a trajectory. Groups the
 * timestamp, x, and y, columns by object id to determine unique trajectories,
 * then computes the average distance and speed for all routes in each
 * trajectory.
 *
 * @note Assumes object_id, timestamp, x, y presorted by (object_id, timestamp).
 *
 * @param num_trajectories number of trajectories (unique object ids)
 * @param object_id column of object (e.g., vehicle) ids
 * @param x coordinates (in kilometers)
 * @param y coordinates (in kilometers)
 * @param timestamp column of timestamps in any resolution
 * @param mr The optional resource to use for output device memory allocations
 *
 * @throw cuspatial::logic_error If object_id isn't cudf::type_id::INT32
 * @throw cuspatial::logic_error If x and y are different types
 * @throw cuspatial::logic_error If timestamp isn't a cudf::TIMESTAMP type
 * @throw cuspatial::logic_error If object_id, x, y, or timestamp contain nulls
 * @throw cuspatial::logic_error If object_id, x, y, and timestamp are different
 * sizes
 *
 * @return a cuDF table of distances (meters) and speeds (meters/second) whose
 * length is `num_trajectories`, sorted by object_id.
 */
std::unique_ptr<cudf::table> trajectory_distances_and_speeds(
  cudf::size_type num_trajectories,
  cudf::column_view const& object_id,
  cudf::column_view const& x,
  cudf::column_view const& y,
  cudf::column_view const& timestamp,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute the spatial bounding boxes of trajectories. Groups the x, y,
 * and timestamp columns by object id to determine unique trajectories, then
 * computes the minimum bounding box to contain all routes in each trajectory.
 *
 * @note Assumes object_id, timestamp, x, y presorted by (object_id, timestamp).
 *
 * @param num_trajectories number of trajectories (unique object ids)
 * @param object_id column of object (e.g., vehicle) ids
 * @param x coordinates (in kilometers)
 * @param y coordinates (in kilometers)
 * @param mr The optional resource to use for output device memory allocations
 *
 * @throw cuspatial::logic_error If object_id isn't cudf::type_id::INT32
 * @throw cuspatial::logic_error If x and y are different types
 * @throw cuspatial::logic_error If object_id, x, or y contain nulls
 * @throw cuspatial::logic_error If object_id, x, and y are different sizes
 *
 * @return a cudf table of bounding boxes with length `num_trajectories` and
 * four columns:
 * x_min - the minimum x-coordinate of each bounding box in kilometers
 * y_min - the minimum y-coordinate of each bounding box in kilometers
 * x_max - the maximum x-coordinate of each bounding box in kilometers
 * y_max - the maximum y-coordinate of each bounding box in kilometers
 */
std::unique_ptr<cudf::table> trajectory_bounding_boxes(
  cudf::size_type num_trajectories,
  cudf::column_view const& object_id,
  cudf::column_view const& x,
  cudf::column_view const& y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
