/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cuspatial/trajectory.hpp>

namespace cuspatial {
namespace experimental {
namespace detail {

/**
 * @brief Derive trajectories from object ids, points, timestamps.
 *
 * Groups the input cols by the given ids, aggregates id counts, and takes the
 * min timestamp, x, and y values from each group.
 *
 * @param[in] id column of object (e.g., vehicle) ids
 * @param[in] x column of x coordinates relative to a camera origin
 * @param[in] y column of y coordinates relative to a camera origin
 * @param[in] timestamp column of timestamps to sort the grouped results
 * @param[in] mr The optional resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to schedule allocations
 *
 * @return a cudf table with the following three int32 columns:
 *   * trajectory ids - the unique ids from the input object ids column
 *   * trajectory counts - the number of of points in the derived trajectories
 *   * trajectory offsets - the cumulative sum of end positions for each group
 *
 * The table is lexicographically sorted ascending by: id, timestamp, x, and y
 */
std::unique_ptr<cudf::experimental::table> derive_trajectories(
    cudf::column_view const& id, cudf::column_view const& x,
    cudf::column_view const& y, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cuspatial
