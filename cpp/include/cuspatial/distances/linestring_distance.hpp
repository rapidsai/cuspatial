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

#include <memory>

namespace cuspatial {

/**
 * @brief Compute distance between pairs of linestrings
 *
 * @param linestring1_offsets Indices to the start coordinate to the first linestring of the pair
 * @param linestring1_points_x x component for points consisting linestrings 1
 * @param linestring1_points_y y component for points consisting linestrings 1
 * @param linestring2_offsets Indices to the start coordinate to the second linestring of the pair
 * @param linestring2_points_x x component for points consisting linestrings 2
 * @param linestring2_points_y y component for points consisting linestrings 2
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column of shortest distances between the pair of linestrings
 *
 * @throw cuspatial::logic_error if `linestring1_offsets.size() != linestring2_offsets.size()`
 * @throw cuspatial::logic_error if size mismatch between the x, y components of the linestring
 * points.
 * @throw cuspatial::logic_error if any of the point arrays have mismatch types.
 * @throw cuspatial::logic_error if any linestring has less than 2 end points.
 *
 */
std::unique_ptr<cudf::column> pairwise_linestring_distance(
  cudf::device_span<cudf::size_type const> linestring1_offsets,
  cudf::column_view const& linestring1_points_x,
  cudf::column_view const& linestring1_points_y,
  cudf::device_span<cudf::size_type const> linestring2_offsets,
  cudf::column_view const& linestring2_points_x,
  cudf::column_view const& linestring2_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
