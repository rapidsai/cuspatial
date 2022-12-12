/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>

namespace cuspatial {

/**
 * @addtogroup spatial_join
 * @{
 */

/**
 * @brief Search a quadtree for polygon or linestring bounding box intersections.
 *
 * @note 2D coordinates are converted into a 1D Morton code by dividing each x/y by the `scale`:
 * (`(x - min_x) / scale` and `(y - min_y) / scale`).
 * @note `max_depth` should be less than 16, since Morton codes are represented as `uint32_t`. The
 * eventual number of levels may be less than `max_depth` if the number of points is small or
 * `max_size` is large.
 *
 * @param quadtree: cudf table representing a quadtree (key, level, is_internal_node, length,
 * offset).
 * @param bbox: cudf table of bounding boxes as four columns (x_min, y_min, x_max, y_max).
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param x_max The upper-right x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param y_max The upper-right y-coordinate of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth at which to stop testing for intersections.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If the quadtree table is malformed
 * @throw cuspatial::logic_error If the bounding box table is malformed
 * @throw cuspatial::logic_error If scale is less than or equal to 0
 * @throw cuspatial::logic_error If x_min is greater than x_max
 * @throw cuspatial::logic_error If y_min is greater than y_max
 * @throw cuspatial::logic_error If max_depth is less than 1 or greater than 15
 *
 * @return A cudf table with two columns:
 *   - bbox_offset - INT32 column of indices for each polygon/linestring bbox that intersects with
 *                   the quadtree.
 *   - quad_offset - INT32 column of indices for each leaf quadrant intersecting with a
 *                   polygon/linestring bbox.
 */
std::unique_ptr<cudf::table> join_quadtree_and_bounding_boxes(
  cudf::table_view const& quadtree,
  cudf::table_view const& bbox,
  double x_min,
  double x_max,
  double y_min,
  double y_max,
  double scale,
  int8_t max_depth,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Test whether the specified points are inside any of the specified polygons.
 *
 * Uses the table of (polygon, quadrant) pairs returned by
 * `cuspatial::join_quadtree_and_bounding_boxes` to ensure only the points in the same quadrant as
 * each polygon are tested for intersection.
 *
 * This pre-filtering can dramatically reduce number of points tested per polygon, enabling
 * faster intersection-testing at the expense of extra memory allocated to store the quadtree and
 * sorted point_indices.
 *
 * @param poly_quad_pairs cudf table of (polygon, quadrant) index pairs returned by
 * `cuspatial::join_quadtree_and_bounding_boxes`
 * @param quadtree cudf table representing a quadtree (key, level, is_internal_node, length,
 * offset).
 * @param point_indices Sorted point indices returned by `cuspatial::quadtree_on_points`
 * @param point_x x-coordinates of points to test
 * @param point_y y-coordinates of points to test
 * @param poly_offsets Begin indices of the first ring in each polygon (i.e. prefix-sum).
 * @param ring_offsets Begin indices of the first point in each ring (i.e. prefix-sum).
 * @param poly_points_x Polygon point x-coordinates.
 * @param poly_points_y Polygon point y-coordinates.
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If the poly_quad_pairs table is malformed.
 * @throw cuspatial::logic_error If the quadtree table is malformed.
 * @throw cuspatial::logic_error If the number of point indices doesn't match the number of points.
 * @throw cuspatial::logic_error If the number of rings is less than the number of polygons.
 * @throw cuspatial::logic_error If any ring has fewer than three vertices.
 * @throw cuspatial::logic_error If the types of point and polygon vertices are different.
 *
 * @return A cudf table with two columns, where each row represents a point/polygon intersection:
 * polygon_offset - UINT32 column of polygon indices
 *   point_offset - UINT32 column of point indices
 *
 * @note The returned polygon and point indices are offsets into the `poly_quad_pairs` inputs and
 * `point_indices`, respectively.
 *
 **/
std::unique_ptr<cudf::table> quadtree_point_in_polygon(
  cudf::table_view const& poly_quad_pairs,
  cudf::table_view const& quadtree,
  cudf::column_view const& point_indices,
  cudf::column_view const& point_x,
  cudf::column_view const& point_y,
  cudf::column_view const& poly_offsets,
  cudf::column_view const& ring_offsets,
  cudf::column_view const& poly_points_x,
  cudf::column_view const& poly_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Finds the nearest linestring to each point in a quadrant, and computes the distances
 * between each point and linestring.
 *
 * Uses the table of (linestring, quadrant) pairs returned by
 * `cuspatial::join_quadtree_and_bounding_boxes` to ensure distances are computed only for the
 * points in the same quadrant as each linestring.
 *
 * @param linestring_quad_pairs cudf table of (linestring, quadrant) index pairs returned by
 * `cuspatial::join_quadtree_and_bounding_boxes`
 * @param quadtree cudf table representing a quadtree (key, level, is_internal_node, length,
 * offset).
 * @param point_indices Sorted point indices returned by `cuspatial::quadtree_on_points`
 * @param point_x x-coordinates of points to test
 * @param point_y y-coordinates of points to test
 * @param linestring_offsets Begin indices of the first point in each linestring (i.e. prefix-sum)
 * @param linestring_points_x Linestring point x-coordinates
 * @param linestring_points_y Linestring point y-coordinates
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If the linestring_quad_pairs table is malformed.
 * @throw cuspatial::logic_error If the quadtree table is malformed.
 * @throw cuspatial::logic_error If the number of point indices doesn't match the number of points.
 * @throw cuspatial::logic_error If any linestring has fewer than two vertices.
 * @throw cuspatial::logic_error If the types of point and linestring vertices are different.
 *
 * @return A cudf table with three columns, where each row represents a point/linestring pair and
 * the distance between the two:
 *
 *   point_offset      - UINT32 column of point indices
 *   linestring_offset - UINT32 column of linestring indices
 *   distance          - FLOAT or DOUBLE column (based on input point data type) of distances
 *                       between each point and linestring
 *
 * @note The returned point and linestring indices are offsets into the `point_indices` and
 * `linestring_quad_pairs` inputs, respectively.
 *
 **/
std::unique_ptr<cudf::table> quadtree_point_to_nearest_linestring(
  cudf::table_view const& linestring_quad_pairs,
  cudf::table_view const& quadtree,
  cudf::column_view const& point_indices,
  cudf::column_view const& point_x,
  cudf::column_view const& point_y,
  cudf::column_view const& linestring_offsets,
  cudf::column_view const& linestring_points_x,
  cudf::column_view const& linestring_points_y,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial
