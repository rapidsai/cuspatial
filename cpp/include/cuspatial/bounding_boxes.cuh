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

#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuspatial {

/**
 * @addtogroup spatial_relationship
 * @{
 */

/**
 * @brief Compute the spatial bounding boxes of sequences of points.
 *
 * Computes a bounding box around all points within each group (consecutive points with the same
 * ID). This function can be applied to trajectory data, polygon vertices, linestring vertices, or
 * any grouped point data.
 *
 * Before merging bounding boxes, each point may be expanded into a bounding box using an
 * optional @p expansion_radius. The point is expanded to a box with coordinates
 * `(point.x - expansion_radius, point.y - expansion_radius)` and
 * `(point.x + expansion_radius, point.y + expansion_radius)`.
 *
 * @note Assumes Object IDs and points are presorted by ID.
 *
 * @tparam IdInputIt Iterator over object IDs. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam PointInputIt Iterator over points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam BoundingBoxOutputIt Iterator over output bounding boxes. Each element is a tuple of two
 * points representing corners of the axis-aligned bounding box. The type of the points is the same
 * as the `value_type` of PointInputIt. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-writeable.
 *
 * @param ids_first beginning of the range of input object ids
 * @param ids_last end of the range of input object ids
 * @param points_first beginning of the range of input point (x,y) coordinates
 * @param bounding_boxes_first beginning of the range of output bounding boxes, one per trajectory
 * @param expansion_radius radius to add to each point when computing its bounding box.
 * @param stream the CUDA stream on which to perform computations.
 *
 * @return An iterator to the end of the range of output bounding boxes.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename IdInputIt,
          typename PointInputIt,
          typename BoundingBoxOutputIt,
          typename T = iterator_vec_base_type<PointInputIt>>
BoundingBoxOutputIt point_bounding_boxes(IdInputIt ids_first,
                                         IdInputIt ids_last,
                                         PointInputIt points_first,
                                         BoundingBoxOutputIt bounding_boxes_first,
                                         T expansion_radius           = T{0},
                                         rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Compute minimum bounding box for each linestring.
 *
 * @tparam LinestringOffsetIterator Iterator type to linestring offsets. Must meet the requirements
 * of [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam VertexIterator Iterator type to linestring vertices. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam BoundingBoxIterator Iterator type to bounding boxes. Must be writable using data of type
 * `cuspatial::box<T>`. Must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-writeable.
 * @tparam T The coordinate data value type.
 * @tparam IndexT  The offset data value type.
 * @param linestring_offsets_first Iterator to beginning of the range of input polygon offsets.
 * @param linestring_offsets_last Iterator to end of the range of input polygon offsets.
 * @param linestring_vertices_first Iterator to beginning of the range of input polygon vertices.
 * @param linestring_vertices_last Iterator to end of the range of input polygon vertices.
 * @param bounding_boxes_first Iterator to beginning of the range of output bounding boxes.
 * @param expansion_radius Optional radius to expand each vertex of the output bounding boxes.
 * @param stream the CUDA stream on which to perform computations and allocate memory.
 *
 * @return An iterator to the end of the range of output bounding boxes.
 *
 * @pre For compatibility with GeoArrow, the number of linestring offsets
 * `std::distance(linestring_offsets_first, linestring_offsets_last)` should be one more than the
 * number of linestrings. The final offset is not used by this function, but the number of offsets
 * determines the output size.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class LinestringOffsetIterator,
          class VertexIterator,
          class BoundingBoxIterator,
          class T      = iterator_vec_base_type<VertexIterator>,
          class IndexT = iterator_value_type<LinestringOffsetIterator>>
BoundingBoxIterator linestring_bounding_boxes(
  LinestringOffsetIterator linestring_offsets_first,
  LinestringOffsetIterator linestring_offsets_last,
  VertexIterator linestring_vertices_first,
  VertexIterator linestring_vertices_last,
  BoundingBoxIterator bounding_boxes_first,
  T expansion_radius           = T{0},
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Compute minimum bounding box for each polygon.
 *
 * @tparam PolygonOffsetIterator Iterator type to polygon offsets. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam RingOffsetIterator Iterator type to polygon ring offsets. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam VertexIterator Iterator type to polygon vertices. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-readable.
 * @tparam BoundingBoxIterator Iterator type to bounding boxes. Must be writable using data of type
 * `cuspatial::box<T>`. Must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be
 * device-writeable.
 * @tparam T The coordinate data value type.
 * @tparam IndexT  The offset data value type.
 * @param polygon_offsets_first Iterator to beginning of the range of input polygon offsets.
 * @param polygon_offsets_last Iterator to end of the range of input polygon offsets.
 * @param polygon_ring_offsets_first Iterator to beginning of the range of input polygon ring
 *                                   offsets.
 * @param polygon_ring_offsets_last Iterator to end of the range of input polygon ring offsets.
 * @param polygon_vertices_first Iterator to beginning of the range of input polygon vertices.
 * @param polygon_vertices_last Iterator to end of the range of input polygon vertices.
 * @param bounding_boxes_first Iterator to beginning of the range of output bounding boxes.
 * @param expansion_radius Optional radius to expand each vertex of the output bounding boxes.
 * @param stream the CUDA stream on which to perform computations and allocate memory.
 *
 * @return An iterator to the end of the range of output bounding boxes.
 *
 * @pre For compatibility with GeoArrow, the number of polygon offsets
 * `std::distance(polygon_offsets_first, polygon_offsets_last)` should be one more than the number
 * of polygons. The number of ring offsets `std::distance(polygon_ring_offsets_first,
 * polygon_ring_offsets_last)` should be one more than the number of total rings. The
 * final offset in each range is not used by this function, but the number of polygon offsets
 * determines the output size.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class PolygonOffsetIterator,
          class RingOffsetIterator,
          class VertexIterator,
          class BoundingBoxIterator,
          class T      = iterator_vec_base_type<VertexIterator>,
          class IndexT = iterator_value_type<PolygonOffsetIterator>>
BoundingBoxIterator polygon_bounding_boxes(PolygonOffsetIterator polygon_offsets_first,
                                           PolygonOffsetIterator polygon_offsets_last,
                                           RingOffsetIterator polygon_ring_offsets_first,
                                           RingOffsetIterator polygon_ring_offsets_last,
                                           VertexIterator polygon_vertices_first,
                                           VertexIterator polygon_vertices_last,
                                           BoundingBoxIterator bounding_boxes_first,
                                           T expansion_radius           = T{0},
                                           rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/detail/bounding_boxes.cuh>
