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

#include <cudf/column/column_view.hpp>

namespace cuspatial {

/**
 * @ingroup distance
 * @brief Compute pairwise (multi)point-to-(multi)point Cartesian distance
 *
 * Computes the cartesian distance between each pair of the multipoints. If input is
 * a single point column, the offset of the column should be std::nullopt.
 *
 * @param points1_xy Column of xy-coordinates of the first point in each pair
 * @param multipoints1_offset Index to the first point of each multipoint in points1_xy
 * @param points2_xy Column of xy-coordinates of the second point in each pair
 * @param multipoints2_offset Index to the second point of each multipoint in points2_xy
 * @return Column of distances between each pair of input points
 */

std::unique_ptr<cudf::column> pairwise_point_distance(
  std::optional<cudf::device_span<cudf::size_type const>> multipoints1_offset,
  cudf::column_view const& points1_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multipoints2_offset,
  cudf::column_view const& points2_xy,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
