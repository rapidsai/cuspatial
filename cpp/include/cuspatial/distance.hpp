/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <cuspatial/constants.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <optional>

namespace cuspatial {

/**
 * @addtogroup distance
 */

/**
 * @brief Compute haversine distances between points in set A and the corresponding points in set B.
 *
 * https://en.wikipedia.org/wiki/Haversine_formula
 *
 * @param[in]  a_lon: longitude of points in set A
 * @param[in]  a_lat:  latitude of points in set A
 * @param[in]  b_lon: longitude of points in set B
 * @param[in]  b_lat:  latitude of points in set B
 * @param[in] radius: radius of the sphere on which the points reside. default: 6371.0 (aprx. radius
 * of earth in km)
 *
 * @return array of distances for all (a_lon[i], a_lat[i]) and (b_lon[i], b_lat[i]) point pairs
 */
std::unique_ptr<cudf::column> haversine_distance(
  cudf::column_view const& a_lon,
  cudf::column_view const& a_lat,
  cudf::column_view const& b_lon,
  cudf::column_view const& b_lat,
  double const radius                 = EARTH_RADIUS_KM,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief computes Hausdorff distances for all pairs in a collection of spaces
 *
 * https://en.wikipedia.org/wiki/Hausdorff_distance
 *
 * Example in 1D (this function operates in 2D):
 * ```
 * spaces
 * [0 2 5] [9] [3 7]
 *
 * spaces represented as points per space and concatenation of all points
 * [0 2 5 9 3 7] [3 1 2]
 *
 * note: the following matrices are visually separated to highlight the relationship of a pair of
 * points with the pair of spaces from which it is produced
 *
 * cartesian product of all
 * points by pair of spaces     distance between points
 * +----------+----+-------+    +---------+---+------+
 * : 00 02 05 : 09 : 03 07 :    : 0  2  5 : 9 : 3  7 :
 * : 20 22 25 : 29 : 23 27 :    : 2  0  3 : 7 : 1  5 :
 * : 50 52 55 : 59 : 53 57 :    : 5  3  0 : 4 : 2  2 :
 * +----------+----+-------+    +---------+---+------+
 * : 90 92 95 : 99 : 93 97 :    : 9  7  4 : 0 : 6  2 :
 * +----------+----+-------+    +---------+---+------+
 * : 30 32 35 : 39 : 33 37 :    : 3  1  2 : 6 : 0  4 :
 * : 70 72 75 : 79 : 73 77 :    : 7  5  2 : 2 : 4  0 :
 * +----------+----+-------+    +---------+---+------+

 * minimum distance from
 * every point in one           Hausdorff distance is
 * space to any point in        the maximum of the
 * the other space              minimum distances
 * +----------+----+-------+    +---------+---+------+
 * :  0       :  9 :  3    :    : 0       : 9 : 3    :
 * :     0    :  7 :  1    :    :         :   :      :
 * :        0 :  4 :  2    :    :         :   :      :
 * +----------+----+-------+    +---------+---+------+
 * :        4 :  0 :     2 :    :       4 : 0 :    2 :
 * +----------+----+-------+    +---------+---+------+
 * :     1    :  6 :  0    :    :         : 6 : 0    :
 * :        2 :  2 :     0 :    :       2 :   :      :
 * +----------+----+-------+    +---------+---+------+
 *
 * Returns:
 * column: [0 4 2 9 0 6 3 2 0]
 * table_view: [0 4 2] [9 0 6] [3 2 0]
 *
 * ```
 *
 * @param[in] xs: x component of points
 * @param[in] ys: y component of points
 * @param[in] space_offsets: beginning index of each space, plus the last space's end offset.
 *
 * @returns An owning object of the result of the hausdorff distances.
 * A table view containing the split view for each input space.
 *
 * @throw cudf::cuda_error if `xs` and `ys` lengths differ
 * @throw cudf::cuda_error if `xs` and `ys` types differ
 * @throw cudf::cuda_error if `space_offsets` size is less than `xs` and `xy`
 * @throw cudf::cuda_error if `xs`, `ys`, or `space_offsets` has nulls
 *
 * @note Hausdorff distances are asymmetrical
 */
std::pair<std::unique_ptr<cudf::column>, cudf::table_view> directed_hausdorff_distance(
  cudf::column_view const& xs,
  cudf::column_view const& ys,
  cudf::column_view const& space_offsets,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
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

/**
 * @brief Compute pairwise (multi)point-to-(multi)polygon Cartesian distance
 *
 * @param multipoints Geometry column of multipoints
 * @param multipolygons Geometry column of multipolygons
 * @param mr Device memory resource used to allocate the returned column.
 * @return Column of distances between each pair of input geometries, same type as input coordinate
 * types.
 *
 * @throw cuspatial::logic_error if `multipoints` and `multipolygons` has different coordinate
 * types.
 * @throw cuspatial::logic_error if `multipoints` is not a point column and `multipolygons` is not a
 * polygon column.
 * @throw cuspatial::logic_error if input column sizes mismatch.
 */

std::unique_ptr<cudf::column> pairwise_point_polygon_distance(
  geometry_column_view const& multipoints,
  geometry_column_view const& multipolygons,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute shortest distance between pairs of linestrings
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

/**
 * @brief Compute pairwise (multi)linestring-to-(multi)polygon Cartesian distance
 *
 * @param multilinestrings Geometry column of multilinestrings
 * @param multipolygons Geometry column of multipolygons
 * @param mr Device memory resource used to allocate the returned column.
 * @return Column of distances between each pair of input geometries, same type as input coordinate
 * types.
 *
 * @throw cuspatial::logic_error if `multilinestrings` and `multipolygons` have different coordinate
 * types.
 * @throw cuspatial::logic_error if `multilinestrings` is not a linestring column and
 * `multipolygons` is not a polygon column.
 * @throw cuspatial::logic_error if input column sizes mismatch.
 */

std::unique_ptr<cudf::column> pairwise_linestring_polygon_distance(
  geometry_column_view const& multilinestrings,
  geometry_column_view const& multipolygons,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
