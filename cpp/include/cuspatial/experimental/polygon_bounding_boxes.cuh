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
 * @brief Compute minimum bounding box for each polygon.
 *
 * @ingroup spatial_relationship
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

}  // namespace cuspatial

#include <cuspatial/experimental/detail/polygon_bounding_boxes.cuh>
