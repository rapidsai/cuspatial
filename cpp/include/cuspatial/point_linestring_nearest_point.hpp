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
#include <cudf/utilities/span.hpp>

namespace cuspatial {

std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
pairwise_point_linestring_nearest_point(
  cudf::column_view points_x,
  cudf::column_view points_y,
  cudf::device_span<cudf::size_type> linestring_offsets,
  cudf::column_view linestring_points_x,
  cudf::column_view linestring_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> pairwise_point_linestring_nearest_segment_idx(
  cudf::column_view points_x,
  cudf::column_view points_y,
  cudf::device_span<cudf::size_type> linestring_offsets,
  cudf::column_view linestring_points_x,
  cudf::column_view linestring_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::
  tuple<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
  pairwise_point_linestring_nearest_point_segment_idx(
    cudf::column_view points_x,
    cudf::column_view points_y,
    cudf::device_span<cudf::size_type> linestring_offsets,
    cudf::column_view linestring_points_x,
    cudf::column_view linestring_points_y,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
