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
 * @brief Compute shortest distance between pairs of linestrings
 * @ingroup distance
 *
 * The shortest distance between two linestrings is defined as the shortest distance
 * between all pairs of segments of the two linestrings. If any of the segments intersect,
 * the distance is 0. The shortest distance between two multilinestrings is defined as the
 * the shortest distance between all pairs of linestrings of the two multilinestrings.
 *
 * The following example contains 4 pairs of linestrings. The first array is a single linestring
 * array and the second array is a multilinestring array.
 * ```
 * First pair:
 * (0, 1) -> (1, 0) -> (-1, 0)
 * {(1, 1) -> (2, 1) -> (2, 0) -> (3, 0)}
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
 * {(1, 0) -> (1, 1) -> (1, 2), (1, -1) -> (1, -2) -> (1, -3)}
 *
 * The linestrings in the pairs are parallel. Their distance is 1 (point (0, 0) to point (1, 0)).
 *
 * Third pair:
 *
 * (0, 0) -> (2, 2) -> (-2, 0)
 * {(2, 0) -> (0, 2), (0, 2) -> (-2, 0)}
 *
 * The linestrings in the pairs intersect, so their distance is 0.
 *
 * Forth pair:
 *
 * (2, 2) -> (-2, -2)
 * {(1, 1) -> (5, 5) -> (10, 0), (-1, -1) -> (-5, -5) -> (-10, 0)}
 *
 * These linestrings contain colinear and overlapping sections, so
 * their distance is 0.
 *
 * The input of above example is:
 * multilinestring1_geometry_offsets: nullopt
 * linestring1_part_offsets:  {0, 3, 5, 8, 10}
 * linestring1_points_xy:
 * {0, 1, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 2, 2, -2, 0, 2, 2, -2, -2}
 *
 * multilinestring2_geometry_offsets: {0, 1, 3, 5, 7}
 * linestring2_offsets:  {0, 4, 7, 10, 12, 14, 17, 20}
 * linestring2_points_xy: {1, 1, 2, 1, 2, 0, 3, 0, 1, 0, 1, 1, 1, 2, 1, -1, 1, -2, 1, -3, 2, 0, 0,
 * 2, 0, 2, -2, 0, 1, 1, 5, 5, 10, 0, -1, -1, -5, -5, -10, 0}
 *
 * Result: {sqrt(2.0)/2, 1, 0, 0}
 * ```
 *
 * @param multilinestring1_geometry_offsets Beginning and ending indices to each multilinestring in
 * the first multilinestring array.
 * @param linestring1_part_offsets Beginning and ending indices for each linestring in the point
 * array. Because the coordinates are interleaved, the actual starting position for the coordinate
 * of linestring `i` is `2*linestring_part_offsets[i]`.
 * @param linestring1_points_xy Interleaved x, y-coordinates of linestring points.
 * @param multilinestring2_geometry_offsets Beginning and ending indices to each multilinestring in
 * the second multilinestring array.
 * @param linestring2_part_offsets Beginning and ending indices for each linestring in the point
 * array. Because the coordinates are interleaved, the actual starting position for the coordinate
 * of linestring `i` is `2*linestring_part_offsets[i]`.
 * @param linestring2_points_xy Interleaved x, y-coordinates of linestring points.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column of shortest distances between each pair of (multi)linestrings
 *
 * @note If `multilinestring_geometry_offset` is std::nullopt, the input is a single linestring
 * array.
 * @note If any of the linestring contains less than 2 points, the behavior is undefined.
 *
 * @throw cuspatial::logic_error if `linestring1_offsets.size() != linestring2_offsets.size()`
 * @throw cuspatial::logic_error if any of the point arrays have mismatched types.
 * @throw cuspatial::logic_error if any linestring has fewer than 2 points.
 *
 */
std::unique_ptr<cudf::column> pairwise_linestring_distance(
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring1_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring1_part_offsets,
  cudf::column_view const& linestring1_points_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring2_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring2_part_offsets,
  cudf::column_view const& linestring2_points_xy,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace cuspatial
