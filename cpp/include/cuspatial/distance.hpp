/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <rmm/resource_ref.hpp>

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
  double const radius               = EARTH_RADIUS_KM,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

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
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute pairwise (multi)point-to-(multi)point Cartesian distance
 *
 * The distance between a pair of multipoints is the shortest Cartesian distance
 * between any pair of points in the two multipoints.
 *
 * @param points1 First column of (multi)points to compute distances
 * @param points2 Second column of (multi)points to compute distances
 * @return Column of distances between each pair of input points
 *
 * @throw cuspatial::logic_error if `multipoints1` and `multipoints2` sizes differ
 * @throw cuspatial::logic_error if either `multipoints1` or `multipoints2` is not a multipoint
 * column
 * @throw cuspatial::logic_error if `multipoints1` and `multipoints2` coordinate types differ
 */
std::unique_ptr<cudf::column> pairwise_point_distance(
  geometry_column_view const& multipoints1,
  geometry_column_view const& multipoints2,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute pairwise (multi)points-to-(multi)linestrings Cartesian distance
 *
 * The distance between a point and a linestring is defined as the minimum Cartesian distance
 * between the point and any segment of the linestring.
 *
 * @param multipoints Column of multipoints to compute distances
 * @param multilinestrings Column of multilinestrings to compute distances
 * @param mr Device memory resource used to allocate the returned column.
 * @return A column containing the distance between each pair of input (multi)points and
 * (multi)linestrings
 *
 * @throw cuspatial::logic_error if `multipoints` and `multilinestrings` sizes differ
 * @throw cuspatial::logic_error if `multipoints` is not a multipoints column or `multilinestrings`
 * is not a multilinestrings column
 * @throw cuspatial::logic_error if `multipoints` and `multilinestrings` coordinate types differ
 */
std::unique_ptr<cudf::column> pairwise_point_linestring_distance(
  geometry_column_view const& multipoints,
  geometry_column_view const& multilinestrings,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute pairwise (multi)point-to-(multi)polygon Cartesian distance
 *
 * The distance between a point and a polygon is defined as the minimum Cartesian distance between
 * the point and any segment of the polygon. If the any point of the multipoint is contained in the
 * polygon, the distance is 0.
 *
 * @param multipoints Geometry column of multipoints
 * @param multipolygons Geometry column of multipolygons
 * @param mr Device memory resource used to allocate the returned column.
 * @return A column containing the distance between each pair of input (multi)points and
 * (multi)polygons
 *
 * @throw cuspatial::logic_error if `multipoints` and `multipolygons` sizes differ
 * @throw cuspatial::logic_error if `multipoints` is not a point column and `multipolygons` is not a
 * polygon column.
 * @throw cuspatial::logic_error if `multipoints` and `multipolygons` coordinate types differ
 */

std::unique_ptr<cudf::column> pairwise_point_polygon_distance(
  geometry_column_view const& multipoints,
  geometry_column_view const& multipolygons,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute pairwise (multi)linestring-to-(multi)linestring Cartesian distance
 *
 * The distance between a pair of multilinestrings is the shortest Cartesian distance
 * between any pair of segments in the two multilinestrings. If any of the segments intersects,
 * the distance is 0.
 *
 * @param multilinestrings1 First column of multilinestrings to compute distances
 * @param multilinestrings2 Second column of multilinestrings to compute distances
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A column containing the distance between each pair of input (multi)linestrings
 *
 * @throw cuspatial::logic_error if `multilinestrings1` and `multilinestrings2` sizes differ
 * @throw cuspatial::logic_error if either `multilinestrings1` or `multilinestrings2` is not a
 * linestring column.
 * @throw cuspatial::logic_error if `multilinestrings1` and `multilinestrings2` coordinate types
 */
std::unique_ptr<cudf::column> pairwise_linestring_distance(
  geometry_column_view const& multilinestrings1,
  geometry_column_view const& multilinestrings2,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute pairwise (multi)linestring-to-(multi)polygon Cartesian distance
 *
 * The distance between a pair of (multi)linestring and (multipolygon) is the shortest Cartesian
 * distance between any pair of segments in the multilinestring and edges in the multipolygon. If
 * any of the segments intersects, or if any linestring is contained in any polygon, the distance is
 * 0.
 *
 * @param multilinestrings Geometry column of multilinestrings
 * @param multipolygons Geometry column of multipolygons
 * @param mr Device memory resource used to allocate the returned column.
 * @return Column of distances between each pair of input geometries, same type as input coordinate
 * types.
 *
 * @throw cuspatial::logic_error if `multilinestrings` and `multipolygons` sizes differ
 * @throw cuspatial::logic_error if either `multilinestrings` is not a linestrings column or
 * `multipolygons` is not a polygon column.
 * @throw cuspatial::logic_error if `multilinestrings` and `multipolygons` has different coordinate
 * types.
 */
std::unique_ptr<cudf::column> pairwise_linestring_polygon_distance(
  geometry_column_view const& multilinestrings,
  geometry_column_view const& multipolygons,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Compute pairwise (multi)polygon-to-(multi)polygon Cartesian distance
 *
 * The distance between a pair of (multi)polygon and (multi)polygon is the shortest Cartesian
 * distance between any pair of edges in the multipolygons. If any edges intersects, or if any
 * polygon is contained in any other polygon, the distance is 0.
 *
 * @param multipolygons1 Geometry column of the multipolygons to compute distance from
 * @param multipolygons2 Geometry column of the multipolygons to compute distance to
 * @param mr Device memory resource used to allocate the returned column.
 *
 * @return Column of distances between each pair of input geometries, same type as input coordinate
 * types.
 */
std::unique_ptr<cudf::column> pairwise_polygon_distance(
  geometry_column_view const& multipolygons1,
  geometry_column_view const& multipolygons2,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
