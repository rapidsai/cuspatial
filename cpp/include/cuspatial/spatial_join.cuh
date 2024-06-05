/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cuspatial/detail/join/intersection.cuh>
#include <cuspatial/detail/join/traversal.cuh>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/point_quadtree.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <iterator>
#include <utility>

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
 * @param quadtree: Reference to a quadtree created using point_quadtree()
 * @param bounding_boxes_first: start bounding boxes iterator
 * @param bounding_boxes_last: end of bounding boxes iterator
 * @param v_min The lower-left (x, y) corner of the area of interest bounding box.
 * @param scale Scale to apply to each x and y distance from x_min and y_min.
 * @param max_depth Maximum quadtree depth at which to stop testing for intersections.
 * @param stream The CUDA stream on which to perform computations
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @return A pair of UINT32 bounding box and leaf quadrant offset device vectors:
 *   - bbox_offset - indices for each polygon/linestring bbox that intersects with the quadtree.
 *   - quad_offset - indices for each leaf quadrant intersecting with a polygon/linestring bbox.
 *
 * @throw cuspatial::logic_error If scale is less than or equal to 0
 * @throw cuspatial::logic_error If max_depth is less than 1 or greater than 15
 */
template <class BoundingBoxIterator,
          class T = typename cuspatial::iterator_vec_base_type<BoundingBoxIterator>>
std::pair<rmm::device_uvector<uint32_t>, rmm::device_uvector<uint32_t>>
join_quadtree_and_bounding_boxes(
  point_quadtree_ref quadtree,
  BoundingBoxIterator bounding_boxes_first,
  BoundingBoxIterator bounding_boxes_last,
  vec_2d<T> const& v_min,
  T scale,
  int8_t max_depth,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Test whether the specified points are inside any of the specified polygons.
 *
 * Uses the (polygon, quadrant) pairs returned by `cuspatial::join_quadtree_and_bounding_boxes` to
 * ensure only the points in the same quadrant as each polygon are tested for intersection.
 *
 * This pre-filtering can dramatically reduce the number of points tested per polygon, enabling
 * faster intersection testing at the expense of extra memory allocated to store the quadtree and
 * sorted point_indices.
 *
 * @param poly_indices_first iterator to beginning of sequence of polygon indices returned by
 *                           cuspatial::join_quadtree_and_bounding_boxes
 * @param poly_indices_first iterator to end of sequence of polygon indices returned by
 *                           cuspatial::join_quadtree_and_bounding_boxes
 * @param quad_indices_first iterator to beginning of sequence of quadrant indices returned by
 *                           cuspatial::join_quadtree_and_bounding_boxes
 * @param quadtree: Reference to a quadtree created using point_quadtree()
 * @param point_indices_first iterator to beginning of sequence of point indices returned by
 *                            `cuspatial::quadtree_on_points`
 * @param point_indices_last iterator to end of sequence of point indices returned by
 *                            `cuspatial::quadtree_on_points`
 * @param points_first iterator to beginning of sequence of (x, y) points to test
 * @param polygons multipolygon_range of polygons.
 * @param stream The CUDA stream on which to perform computations
 * @param mr The optional resource to use for output device memory allocations.
 *
 * @throw cuspatial::logic_error If the number of rings is less than the number of polygons.
 * @throw cuspatial::logic_error If any ring has fewer than four vertices.
 * @throw cuspatial::logic_error if the number of multipolygons does not equal the total number of
 *        multipolygons (one polygon per multipolygon)
 *
 * @return A pair of rmm::device_uvectors where each row represents a point/polygon intersection:
 *     polygon_offset - uint32_t polygon indices
 *     point_offset   - uint32_t point indices
 *
 * @note Currently only supports single-polygon multipolygons.
 * @note The returned polygon and point indices are offsets into the `poly_quad_pairs` input range
 *       and `point_indices` range, respectively.
 *
 **/
template <class PolyIndexIterator,
          class QuadIndexIterator,
          class PointIndexIterator,
          class PointIterator,
          class MultiPolygonRange,
          class IndexType = iterator_value_type<PointIndexIterator>>
std::pair<rmm::device_uvector<IndexType>, rmm::device_uvector<IndexType>> quadtree_point_in_polygon(
  PolyIndexIterator poly_indices_first,
  PolyIndexIterator poly_indices_last,
  QuadIndexIterator quad_indices_first,
  point_quadtree_ref quadtree,
  PointIndexIterator point_indices_first,
  PointIndexIterator point_indices_last,
  PointIterator points_first,
  MultiPolygonRange polygons,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Finds the nearest linestring to each point in a quadrant, and computes the distances
 * between each point and linestring.
 *
 * Uses the (linestring, quadrant) pairs returned by `cuspatial::join_quadtree_and_bounding_boxes`
 * to ensure distances are computed only for the points in the same quadrant as each linestring.
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
template <class LinestringIndexIterator,
          class QuadIndexIterator,
          class PointIndexIterator,
          class PointIterator,
          class MultiLinestringRange,
          typename IndexType = iterator_value_type<PointIndexIterator>,
          typename T         = iterator_vec_base_type<PointIterator>>
std::tuple<rmm::device_uvector<IndexType>, rmm::device_uvector<IndexType>, rmm::device_uvector<T>>
quadtree_point_to_nearest_linestring(
  LinestringIndexIterator linestring_indices_first,
  LinestringIndexIterator linestring_indices_last,
  QuadIndexIterator quad_indices_first,
  point_quadtree_ref quadtree,
  PointIndexIterator point_indices_first,
  PointIndexIterator point_indices_last,
  PointIterator points_first,
  MultiLinestringRange linestrings,
  rmm::cuda_stream_view stream      = rmm::cuda_stream_default,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace cuspatial

#include <cuspatial/detail/join/quadtree_bbox_filtering.cuh>
#include <cuspatial/detail/join/quadtree_point_in_polygon.cuh>
#include <cuspatial/detail/join/quadtree_point_to_nearest_linestring.cuh>
