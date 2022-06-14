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
 * @ingroup distance
 * @brief Compute shortest distance between pairs of point to linestrings (a.k.a. polylines)
 *
 * @param points_x x-components of points in the point of each pair
 * @param points_y y-component of points in the point of each pair
 * @param linestring_offsets Indices of the first point of the linestring of each pair
 * @param linestring_points_x x-components of points in the linestring of each pair
 * @param linestring_points_y y-component of points in the linestring of each pair
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column of shortest distances between each pair of linestrings
 *
 * @note If any of the linestring contains less than 2 points, the behavior is undefined.
 *
 * @throw cuspatial::logic_error if there is a size mismatch between the x- and y-coordinates of the
 * linestring points.
 * @throw cuspatial::logic_error if any of the point arrays have mismatched types.
 * @throw cuspatial::logic_error if any linestring has fewer than 2 points.
 *
 */
std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  cudf::column_view const& points_x,
  cudf::column_view const& points_y,
  cudf::device_span<cudf::size_type const> linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
