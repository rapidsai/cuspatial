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
 * @brief Compute the total distance (in meters) and average speed (in m/s) of objects in
 * trajectories.
 *
 * @note Assumes object_id, timestamp, x, y presorted by (object_id, timestamp).
 *
 * @tparam IdInputIt Iterator over object IDs. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam PointInputIt Iterator over points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam TimestampInputIt Iterator over timestamps. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam OutputIt Iterator over output (distance, speed) pairs. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-writeable.
 * @tparam IndexT The type of the object IDs.
 *
 * @param num_trajectories number of trajectories (unique object ids)
 * @param ids_first beginning of the range of input object ids
 * @param ids_last end of the range of input object ids
 * @param points_first beginning of the range of input point (x,y) coordinates
 * @param timestamps_first beginning of the range of input timestamps
 * @param distances_and_speeds_first beginning of the range of output (distance, speed) pairs
 * @param stream the CUDA stream on which to perform computations and allocate memory.
 *
 * @return An iterator to the end of the range of output (distance, speed) pairs.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
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
