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
    cudf::column_view const& object_id, cudf::column_view const& x,
    cudf::column_view const& y, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @copydoc cudf::experimental::trajectory_bounding_boxes()
 * @param stream Optional CUDA stream on which to schedule allocations
 */
std::unique_ptr<cudf::experimental::table> trajectory_bounding_boxes(
    cudf::column_view const& object_id, cudf::column_view const& x,
    cudf::column_view const& y,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Count the number of unique object ids
 *
 * @param object_id Column of object (e.g., vehicle) ids
 * @param stream Optional CUDA stream on which to schedule allocations
 * @return cudf::size_type The number of unique elements
 */
inline cudf::size_type count_unique_ids(cudf::column_view const& object_id,
                                        cudaStream_t stream = 0) {
  auto policy = rmm::exec_policy(stream);
  rmm::device_vector<int32_t> unique_keys(object_id.size());
  auto last_key_pos =
      thrust::unique_copy(policy->on(stream), object_id.begin<int32_t>(),
                          object_id.end<int32_t>(), unique_keys.begin());
  return thrust::distance(unique_keys.begin(), last_key_pos);
}

}  // namespace detail
}  // namespace experimental
}  // namespace cuspatial
