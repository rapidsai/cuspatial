/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuspatial {

/**
 * @addtogroup trajectory_api
 * @{
 */

/**
 * @brief Compute the distance and speed of objects in trajectories. Groups the
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
template <typename IdInputIt,
          typename PointInputIt,
          typename TimestampInputIt,
          typename OutputIt,
          typename IndexT = iterator_value_type<IdInputIt>>
OutputIt trajectory_distances_and_speeds(IndexT num_trajectories,
                                         IdInputIt ids_first,
                                         IdInputIt ids_last,
                                         PointInputIt points_first,
                                         TimestampInputIt timestamps_first,
                                         OutputIt distances_and_speeds_first,
                                         rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include "detail/trajectory_distances_and_speeds.cuh"
