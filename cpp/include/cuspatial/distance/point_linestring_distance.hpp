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

#include <cudf/column/column_view.hpp>

namespace cuspatial {

/**
 * @brief Compute distance between pairs of points and linestrings (a.k.a. polylines)
 *
 * The distance between a point and a linestring is defined as the minimum distance
 * between the point and any segment of the linestring. For each input point, this
 * function returns the distance between the point and the corresponding linestring.
 *
 * The following example contains 2 pairs of points and linestrings.
 * ```
 * First pair:
 * Point: (0, 0)
 * Linestring: (0, 1) -> (1, 0) -> (2, 0)
 *
 * Second pair:
 * Point: (1, 1)
 * Linestring: (0, 0) -> (1, 1) -> (2, 0) -> (3, 0) -> (3, 1)
 *
 * The input of the abbove example is:
 * points_x: {0, 1}
 * points_y: {0, 1}
 * linestring_offsets: {0, 3}
 * linestring_x: {0, 1, 2, 0, 1, 2, 3, 3}
 * linestring_y: {1, 0, 0, 0, 1, 0, 0, 1}
 *
 * Result: {sqrt(2)/2, 0}
 * ```
 *
 * @param points_x x-coordinates of points.
 * @param points_y y-coordinates of points.
 * @param linestring_offsets Indices of the start of each linestring in the `linestring_x` and
 * `linestring_y` arrays.
 * @param linestring_points_x x-coordinates of linestring points.
 * @param linestring_points_y y-coordinates of linestring points.
 * @param mr Device memory resource used to allocate the returned column.
 * @return A column containing the distance between each pair of corresponding points and
 * linestrings.
 *
 * @throws cuspatial::logic_error if the number of points and linestrings do not match.
 * @throws cuspatial::logic_error if there is a size mismatch between the x- and y-coordinates of
 * the points or linestring points.
 * @throws cuspatial::logic_error if the any of the point arrays have mismatched types.
 */
std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  cudf::column_view const& points_x,
  cudf::column_view const& points_y,
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
