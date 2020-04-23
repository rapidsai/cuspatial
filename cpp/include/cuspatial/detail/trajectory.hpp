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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/unique.h>

#include <cudf/column/column_view.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/trajectory.hpp>

namespace cuspatial {
namespace experimental {
namespace detail {

/**
 * @copydoc cudf::experimental::derive_trajectories()
 * @param stream Optional CUDA stream on which to schedule allocations
 */
std::pair<std::unique_ptr<cudf::experimental::table>,
          std::unique_ptr<cudf::column>>
derive_trajectories(
    cudf::column_view const& object_id, cudf::column_view const& x,
    cudf::column_view const& y, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @copydoc cudf::experimental::trajectory_distances_and_speeds()
 * @param stream Optional CUDA stream on which to schedule allocations
 */
std::unique_ptr<cudf::experimental::table> trajectory_distances_and_speeds(
    cudf::size_type num_trajectories, cudf::column_view const& object_id,
    cudf::column_view const& x, cudf::column_view const& y,
    cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @copydoc cudf::experimental::trajectory_bounding_boxes()
 * @param stream Optional CUDA stream on which to schedule allocations
 */
std::unique_ptr<cudf::experimental::table> trajectory_bounding_boxes(
    cudf::size_type num_trajectories, cudf::column_view const& object_id,
    cudf::column_view const& x, cudf::column_view const& y,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cuspatial
