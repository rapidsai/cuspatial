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

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup distance
 * @{
 */

/**
 * @brief Compute shortest distance between pairs of linestrings
 *
 * The shortest distance between two linestrings is defined as the shortest distance
 * between all pairs of segments of the two linestrings. If any of the segments intersect,
 * the distance is 0.
 *
 * The following example contains 4 pairs of linestrings.
 * ```
 * First pair:
 * (0, 1) -> (1, 0) -> (-1, 0)
 * (1, 1) -> (2, 1) -> (2, 0) -> (3, 0)
 *
 *     |
 *     *   #---#
 *     | \     |
 * ----O---*---#---#
 *     | /
 *     *
 *     |
 *
 * The shortest distance between the two linestrings is the distance
 * from point (1, 1) to segment (0, 1) -> (1, 0), which is sqrt(2)/2.
 *
 * Second pair:
 *
 * (0, 0) -> (0, 1)
 * (1, 0) -> (1, 1) -> (1, 2)
 *
 * These linestrings are parallel. Their distance is 1 (point (0, 0) to point (1, 0)).
 *
 * Third pair:
 *
 * (0, 0) -> (2, 2) -> (-2, 0)
 * (2, 0) -> (0, 2)
 *
 * These linestrings intersect, so their distance is 0.
 *
 * Forth pair:
 *
 * (2, 2) -> (-2, -2)
 * (1, 1) -> (5, 5) -> (10, 0)
 *
 * These linestrings contain colinear and overlapping sections, so
 * their distance is 0.
 *
 * The input of above example is:
 * linestring1_offsets:  {0, 3, 5, 8}
 * linestring1_points_x: {0, 1, -1, 0, 0, 0, 2, -2, 2, -2}
 * linestring1_points_y: {1, 0, 0, 0, 1, 0, 2, 0, 2, -2}
 * linestring2_offsets:  {0, 4, 7, 9}
 * linestring2_points_x: {1, 2, 2, 3, 1, 1, 1, 2, 0, 1, 5, 10}
 * linestring2_points_y: {1, 1, 0, 0, 0, 1, 2, 0, 2, 1, 5, 0}
 *
 * Result: {sqrt(2.0)/2, 1, 0, 0}
 * ```
 *
 * @param linestring1_offsets Indices of the first point of the first linestring of each pair
 * @param linestring1_points_x x-components of points in the first linestring of each pair
 * @param linestring1_points_y y-component of points in the first linestring of each pair
 * @param linestring2_offsets Indices of the first point of the second linestring of each pair
 * @param linestring2_points_x x-component of points in the second linestring of each pair
 * @param linestring2_points_y y-component of points in the second linestring of each pair
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column of shortest distances between each pair of linestrings
 *
 * @note If any of the linestring contains less than 2 points, the behavior is undefined.
 *
 * @throw cuspatial::logic_error if `linestring1_offsets.size() != linestring2_offsets.size()`
 * @throw cuspatial::logic_error if there is a size mismatch between the x- and y-coordinates of the
 * linestring points.
 * @throw cuspatial::logic_error if any of the point arrays have mismatched types.
 * @throw cuspatial::logic_error if any linestring has fewer than 2 points.
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

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
