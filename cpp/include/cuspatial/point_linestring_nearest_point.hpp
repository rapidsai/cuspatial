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

#include <thrust/optional.h>

#include <tuple>

namespace cuspatial {

/**
 * @brief Compute the nearest points and geometry id between a pair of (multi)point and
 * (multi)linestring
 *
 * The nearest point from a test point to a linestring is a point on the linestring that has
 * the shortest distance to the test point compared to any other points on the linestring.
 *
 * The nearest point from a test multipoint to a multilinestring is the nearest point that
 * has the shortest distance in all pairs of points and linestrings.
 *
 * In addition, this API returns the geometry and part ID where the nearest point locates.
 * - The point id in a multipoint indicates which point is the nearest point.
 * - The linestring id in a multilinestring indicates which linestring the nearest point locates
 * - The segment id in a linestring indicates which segment the nearest point locates. It is
 *   the same as the id to the starting point of the segment.
 *
 * @note When the test point is not a multipoint, the nearest point id is omitted.
 * When the test linestring is not a multilinestring, the nearest linestring id is omitted.
 *
 * The below example computes the nearest point from 2 points to 2 linestrings:
 *
 * The first pair:
 * Point: (0.0, 0.0)
 * Linestring: (1.0, -1.0) -> (1.0, 0.0) -> (0.0, 1.0)
 * Nearest segment id in linestring: 1
 * Nearest Point on LineString Coordinate: (0.5, 0.5)
 *
 * The second pair:
 * Point: (1.0, 2.0)
 * Linestring: (0.0, 0.0) -> (3.0, 1.0) -> (3.9, 4) -> (5.5, 1.2)
 * Nearest segment id in linestring: 0
 * Nearest Point on LineString Coordinate: (1.5, 0.5)
 *
 * Input:
 * multipoint_parts_offsets: std::nullopt
 * points_xy: {0.0, 0.0}
 * multilinestring_parts_offsets: std::nullopt
 * linestring_offsets:  {0, 3, 7}
 * linestring_points_xy: {1, -1, 1, 0, 0, 1, 0, 0, 3, 1, 3.9, 4, 5.5, 1.2}
 *
 * Output:
 * std::tuple(
 *   std::nullopt,
 *   std::nullopt,
 *   {1, 0},
 *   {0.5, 0.5, 1.5, 0.5}
 * )
 *
 * The below example computes the nearest point from 3 multipoints to 3 multilinestrings:
 *
 * The first pair:
 * MultiPoint: {(1.1, 3.0), (3.6, 2.4)}
 * MultiLineString: {(2.1, 3.14) -> (8.4, -0.5) -> (6.0, 1.4), (-1.0, 0.0) -> (-1.7, 0.83)}
 * Nearest Point id in MultiPoint: 1
 * Nearest LineString id in MultiLineString: 0
 * Nearest Segment id in LineString: 0
 * Nearest Point on LineString Coordinate: (3.54513, 2.30503)
 *
 * The second pair:
 * MultiPoint: {(10.0, 15.0)}
 * MultiLineString: {(20.14, 13.5) -> (18.3, 14.3), (8.34, 9.1) -> (9.9, 9.4)}
 * Nearest Point id in MultiPoint: 0
 * Nearest LineString id in MultiLineString: 1
 * Nearest Segment id in LineString: 0
 * Nearest Point in LineString Coordinate: (9.9, 9.4)
 *
 * The third pair:
 * MultiPoint: {(-5.0, -8.7), (-6.28, -7.0), (-10.0, -10.0)}
 * MultiLineString: {(-20.0, 0.0) -> (-15.0, -15.0) -> (0.0, -18.0) -> (0.0, 0.0)}
 * Nearest Point id in MultiPoint: 0
 * Nearest LineString id in MultiLineString: 0
 * Nearest Segment id in LineString: 2
 * Nearest Point in LineString Coordinate: (0.0, -8.7)
 *
 * Input:
 * multipoint_parts_offsets: {0, 2, 3, 6}
 * points_xy: {1.1, 3.0, 3.6, 2.4, 10.0, 15.0, -5.0, -8.7, -6.28, -7.0, -10.0, -10.0}
 * multilinestring_parts_offsets: {0, 2, 4, 5}
 * linestring_offsets:  {0, 3, 5, 7, 9, 13}
 * linestring_points_xy: {
 *   2.1, 3.14, 8.4, -0.5, 6.0, 1.4, -1.0, 0.0,
 *   -1.7, 0.83, 20.14, 13.5, 18.3, 14.3, 8.34, 9.1,
 *   9.9, 9.4, -20.0, 0.0, -15.0, -15.0, 0.0, -18.0,
 *   0.0, 0.0
 * }
 *
 * Output:
 * std::tuple(
 *  {1, 0, 0},
 *  {0, 1, 0},
 *  {0, 0, 2},
 *  {3.545131432802666, 2.30503517215846, 9.9, 9.4, 0.0, -8.7}
 * )
 *
 * @param multipoint_geometry_offsets Beginning and ending indices for each multipoint
 * @param points_xy Interleaved x, y-coordinate of points
 * @param multilinestring_geometry_offsets Beginning and ending indices for each multilinestring
 * @param linestring_part_offsets Beginning and ending indices for each linestring
 * @param linestring_points_xy Interleaved x, y-coordinates of the linestring points
 * @param mr Device memory resource used to allocate the returned column.
 * @return A tuple containing the following:
 * 1. The point id of the nearest point in multipoint (std::nullopt if input is not multipoint)
 * 2. The linestring id where the nearest point in multilinestring locates (std::nullopt if input is
 * not multlinestring)
 * 3. The segment id where the nearest point in the (multi)linestring locates
 * 4. The interleaved x, y-coordinate of the nearest point on the (multi)linestring
 *
 * @throws cuspatial::logic_error if `points_xy` or `linestring_points_xy` contains odd number of
 * coordinates.
 * @throws cuspatial::logic_error if `points_xy` or `linestring_points_xy` are not floating point
 * type.
 * @throws cuspatial::logic_error if the number of (multi)point(s) mismatch the number of
 * (multi)linestring(s).
 * @throws cuspatial::logic_error if the type of `point_xy` and `linestring_points_xy` mismatch.
 * @throws cuspatial::logic_error if any of `point_xy` and `linestring_points_xy` contains null.
 */
std::tuple<std::optional<std::unique_ptr<cudf::column>>,
           std::optional<std::unique_ptr<cudf::column>>,
           std::unique_ptr<cudf::column>,
           std::unique_ptr<cudf::column>>
pairwise_point_linestring_nearest_points(
  std::optional<cudf::device_span<cudf::size_type const>> multipoint_geometry_offsets,
  cudf::column_view points_xy,
  std::optional<cudf::device_span<cudf::size_type const>> multilinestring_geometry_offsets,
  cudf::device_span<cudf::size_type const> linestring_part_offsets,
  cudf::column_view linestring_points_xy,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial
