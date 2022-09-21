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

#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <iterator>
#include <memory>

namespace cuspatial {

/**
 * @addtogroup trajectory_api
 * @{
 */

/**
 * @brief Derive trajectories from object ids, points, and timestamps.
 *
 * Output points and timestamps are reordered to be grouped by object ID and ordered by timestamp
 * within groups.  Returns a vector containing the offset index of the first object of each
 * trajectory in the output.
 *
 * @tparam IdInputIt Iterator over object IDs. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam PointInputIt Iterator over points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam TimestampInputIt Iterator over timestamps. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam IdOutputIt Iterator over output object IDs. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-writeable.
 * @tparam PointOutputIt Iterator over output points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-writeable.
 * @tparam TimestampOutputIt Iterator over output timestamps. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-writeable.
 *
 * @param ids_first beginning of the range of input object ids
 * @param ids_last end of the range of input object ids
 * @param points_first beginning of the range of input point (x,y) coordinates
 * @param timestamps_first beginning of the range of input timestamps
 * @param ids_out_first beginning of the range of output object ids
 * @param points_out_first beginning of the range of output point (x,y) coordinates
 * @param timestamps_out_first beginning of the range of output timestamps
 * @param stream the CUDA stream on which to perform computations and allocate memory.
 * @param mr optional resource to use for output device memory allocations
 *
 * @return a unique_ptr to a device_uvector containing the offset index of the first object of each
 * trajectory in the sorted output. These offsets can be used to access the sorted output data.
 *
 * @pre There must be no overlap between any of the input and output ranges.
 * @pre The type of the object IDs and timestamps must support strict weak ordering via comparison
 *      operators.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename IdInputIt,
          typename PointInputIt,
          typename TimestampInputIt,
          typename IdOutputIt,
          typename PointOutputIt,
          typename TimestampOutputIt,
          typename OffsetType = std::int32_t>
std::unique_ptr<rmm::device_uvector<OffsetType>> derive_trajectories(
  IdInputIt ids_first,
  IdInputIt ids_last,
  PointInputIt points_first,
  TimestampInputIt timestamps_first,
  IdOutputIt ids_output_first,
  PointOutputIt points_output_first,
  TimestampOutputIt timestamps_output_first,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/experimental/detail/derive_trajectories.cuh>
