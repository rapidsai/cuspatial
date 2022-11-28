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
 * @brief Compute distance between pairs of points and linestrings
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
 * The input of the above example is:
 * multipoint_geometry_offsets: nullopt
 * points_xy: {0, 1, 0, 1}
 * multilinestring_geometry_offsets: nullopt
 * linestring_part_offsets: {0, 3, 8}
 * linestring_xy: {0, 1, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0, 3, 0, 3, 1}
 *
 * Result: {sqrt(2)/2, 0}
 * ```
 *
 * The following example contains 3 pairs of MultiPoint and MultiLinestring.
 * ```
 * First pair:
 * MultiPoint: (0, 1)
 * MultiLinestring: (0, -1) -> (-2, -3), (-4, -5) -> (-5, -6)
 *
 * Second pair:
 * MultiPoint: (2, 3), (4, 5)
 * MultiLinestring: (7, 8) -> (8, 9)
 *
 * Third pair:
 * MultiPoint: (6, 7), (8, 9)
 * MultiLinestring: (9, 10) -> (10, 11)

 * The input of the above example is:
 * multipoint_geometry_offsets: {0, 1, 3, 5}
 * points_xy: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
 * multilinestring_geometry_offsets: {0, 2, 3, 5}
 * linestring_part_offsets: {0, 2, 4, 6, 8}
 * linestring_points_xy: {0, -1, -2, -3, -4, -5, -5, -6, 7, 8, 8, 9, 9, 10, 10 ,11}
 *
 * Result: {2.0, 4.24264, 1.41421}
 * ```
 *
 * @param multipoint_geometry_offsets Beginning and ending indices to each geometry in the
 * multi-point
 * @param points_xy Interleaved x, y-coordinates of points
 * @param multilinestring_geometry_offsets Beginning and ending indices to each geometry in the
 * multi-linestring
 * @param linestring_part_offsets Beginning and ending indices for each linestring in the point
 * array. Because the coordinates are interleaved, the actual starting position for the coordinate
 * of linestring `i` is `2*linestring_part_offsets[i]`.
 * @param linestring_points_xy Interleaved x, y-coordinates of linestring points.
 * @param mr Device memory resource used to allocate the returned column.
 * @return A column containing the distance between each pair of corresponding points and
 * linestrings.
 *
 * @note Any optional geometry indices, if is `nullopt`, implies the underlying geometry contains
 * only one component. Otherwise, it contains multiple components.
 *
 * @throws cuspatial::logic_error if the number of (multi)points and (multi)linestrings do not
 * match.
 * @throws cuspatial::logic_error if the any of the point arrays have mismatched types.
 */
std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
  cudf::column_view const& points_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring_part_offsets,
  cudf::column_view const& linestring_points_xy,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
