/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial/column/geometry_column_view.hpp>

#include <cudf/column/column.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cuspatial {
/**
 * @brief Result of linestring intersections
 *
 * Owning object to hold the result of linestring intersections.
 * The results are modeled after arrow type List<Union<Point, LineString>>,
 * with additional information about the indices where the intersection locates.
 */
struct linestring_intersection_column_result {
  /// List offsets to the union column
  std::unique_ptr<cudf::column> geometry_collection_offset;

  /// Union Column Results
  std::unique_ptr<cudf::column> types_buffer;
  std::unique_ptr<cudf::column> offset_buffer;

  /// Child 0: Point Results as List Type Column
  std::unique_ptr<cudf::column> points;

  /// Child 1: Segment Results as List Type Column
  std::unique_ptr<cudf::column> segments;

  /// Look-back Indices
  std::unique_ptr<cudf::column> lhs_linestring_id;
  std::unique_ptr<cudf::column> lhs_segment_id;
  std::unique_ptr<cudf::column> rhs_linestring_id;
  std::unique_ptr<cudf::column> rhs_segment_id;
};

linestring_intersection_column_result pairwise_linestring_intersection(
  geometry_column_view const& multilinestrings_lhs,
  geometry_column_view const& multilinestrings_rhs,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
