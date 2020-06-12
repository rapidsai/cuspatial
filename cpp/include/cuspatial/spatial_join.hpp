/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <memory>

namespace cuspatial {

/**
 * @brief pair quadtree quadrants and polygons by intersection tests of quadrants and polygon bboxes
 *
 * @param quadtree: cudf table representing a quadtree points index (key, level, is_quad, length,
 * offset)
 * @param poly_bbox: cudf table of bounding boxes as four columns (x_min, y_min, x_max, y_max)
 * @param x_min The lower-left x-coordinate of the area of interest bounding box.
 * @param x_max The upper-right x-coordinate of the area of interest bounding box.
 * @param y_min The lower-left y-coordinate of the area of interest bounding box.
 * @param y_max The upper-right y-coordinate of the area of interest bounding box.
 * @param scale: grid cell size along both x and y dimensions.
 * scale works with x1 and x2 to convert x/y coodiantes into a Morton code in 2D space
 * @param max_depth: largest quadtree depth. the value should be less than 16 as uint32_t is used
 * for Morton code representation the actual number of levels may be less than max_depth when #of
 * points are small and/or min_size (next) is large
 *
 * @return array of (polygon_index, quadrant_index) pairs that quadrant intersects with polygon bbox
 * quadrant_index and polygon_index are offsets of quadrant and polygon arrays, respectively
 */

std::unique_ptr<cudf::table> quad_bbox_join(
  cudf::table_view const &quadtree,
  cudf::table_view const &poly_bbox,
  double x_min,
  double y_min,
  double x_max,
  double y_max,
  double scale,
  cudf::size_type max_depth,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief pair points and polygons using ray-cast based point-in-polygon test algorithm in two
 *phases: phase 1 counts the total number of output pairs for precise memory allocation phase 2
 *actually writes (point,polygon) pairs
 *
 * @param pq_pair: table of two arrays for (quadrant,polygon) pairs derived from spatial filtering
 * @param quadtree: table of five arrays derived from quadtree indexing on points (key,lev,sign,
 * length, fpos)
 * @param pnt: table of two arrays for points (x,y). note that points are in-place sorted in
 * quadtree construction and have different orders than the orginal input points.
 * @param fpos: feature/polygon offset array to rings
 * @param rpos: ring offset array to vertex
 * @param poly_x: polygon x coordiante array.
 * @param poly_y: polygon y coordiante array.
 *
 * @return array of (polygon_index, point_index) pairs that point is within polyon;
 * point_index and polygon_index are offsets of point and polygon arrays, respectively
 */
// std::unique_ptr<cudf::table> pip_refine(
//   cudf::table_view const& pq_pair,
//   cudf::table_view const& quadtree,
//   cudf::table_view const& pnt,
//   cudf::column_view const& poly_fpos,
//   cudf::column_view const& poly_rpos,
//   cudf::column_view const& poly_x,
//   cudf::column_view const& poly_y,
//   rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace cuspatial
