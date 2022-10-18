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

namespace cuspatial {

/**
 * @brief Compute the spatial bounding boxes of trajectories.
 *
 * Computes a tight bounding box around all points within each trajectory (points with the same ID).
 *
 * @note Assumes Object IDs and points are presorted by ID. This can be done using
 * cuspatial::derive_trajectories.
 *
 * @tparam IdInputIt Iterator over object IDs. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam PointInputIt Iterator over points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam PointOutputIt Iterator over output bounding box points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-writeable.
 *
 * @param ids_first beginning of the range of input object ids
 * @param ids_last end of the range of input object ids
 * @param points_first beginning of the range of input point (x,y) coordinates
 * @param bounding_box_minima_first beginning of the range of output minimum bounding box
 *                                  coordinates, one per trajectory
 * @param bounding_box_maxima_first beginning of the range of output maximum bounding box
 *                                  coordinates, one per trajectory
 * @param stream the CUDA stream on which to perform computations.
 *
 * @return A `std::pair` of iterators to the ends of the ranges of output minimum and maximum
 *         bounding box coordinates.
 */
template <typename IdInputIt, typename PointInputIt, typename PointOutputIt>
std::pair<PointOutputIt, PointOutputIt> trajectory_bounding_boxes(
  IdInputIt ids_first,
  IdInputIt ids_last,
  PointInputIt points_first,
  PointOutputIt bounding_box_minima_first,
  PointOutputIt bounding_box_maxima_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include "detail/trajectory_bounding_boxes.cuh"
