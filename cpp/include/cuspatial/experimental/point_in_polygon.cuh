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

#include <rmm/cuda_stream_view.hpp>

namespace cuspatial {

/**
 * @ingroup spatial_relationship
 *
 * @brief Tests whether the specified points are inside any of the specified polygons.
 *
 * Tests whether points are inside at most 31 polygons. Polygons are a collection of one or more
 * rings. Rings are a collection of three or more vertices.
 *
 * Each input point will map to one `int32_t` element in the output. Each bit (except the sign bit)
 * represents a hit or miss for each of the input polygons in least-significant-bit order. i.e.
 * `output[3] & 0b0010` indicates a hit or miss for the 3rd point against the 2nd polygon.
 *
 *
 * @tparam Cart2dItA iterator type for point array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Cart2dItB iterator type for point array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorA iterator type for offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorB iterator type for offset array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt iterator type for output array. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI], be device-accessible, mutable and
 * iterate on `int32_t` type.
 *
 * @param test_points_first begin of range of test points
 * @param test_points_last end of range of test points
 * @param polygon_offsets_first begin of range of indices to the first ring in each polygon
 * @param polygon_offsets_last end of range of indices to the first ring in each polygon
 * @param ring_offsets_first begin of range of indices to the first point in each ring
 * @param ring_offsets_last end of range of indices to the first point in each ring
 * @param polygon_points_first begin of range of polygon points
 * @param polygon_points_last end of range of polygon points
 * @param output begin iterator to the output buffer
 * @param stream The CUDA stream to use for kernel launches.
 * @return iterator to one past the last element in the output buffer
 *
 * @note Limit 31 polygons per call. Polygons may contain multiple rings.
 * @note Direction of rings does not matter.
 * @note This algorithm supports the ESRI shapefile format, but assumes all polygons are "clean" (as
 * defined by the format), and does _not_ verify whether the input adheres to the shapefile format.
 * @note The points of the rings can be either explicitly closed (the first and last vertex
 * overlaps), or implicitly closed (not overlaps). Either input format is supported.
 * @note Overlapping rings negate each other. This behavior is not limited to a single negation,
 * allowing for "islands" within the same polygon.
 * @note `poly_ring_offsets` must contain only the rings that make up the polygons indexed by
 * `poly_offsets`. If there are rings in `poly_ring_offsets` that are not part of the polygons in
 * `poly_offsets`, results are likely to be incorrect and behavior is undefined.
 *
 * ```
 *   poly w/two rings         poly w/four rings
 * +-----------+          +------------------------+
 * :███████████:          :████████████████████████:
 * :███████████:          :██+------------------+██:
 * :██████+----:------+   :██:  +----+  +----+  :██:
 * :██████:    :██████:   :██:  :████:  :████:  :██:
 * +------;----+██████:   :██:  :----:  :----:  :██:
 *        :███████████:   :██+------------------+██:
 *        :███████████:   :████████████████████████:
 *        +-----------+   +------------------------+
 * ```
 *
 * @pre All point iterators must have the same `vec_2d` value type, with  the same underlying
 * floating-point coordinate type (e.g. `cuspatial::vec_2d<float>`).
 * @pre All offset iterators must have the same integral value type.
 * @pre Output iterator must be mutable and iterate on int32_t type.
 *
 * @throw cuspatial::logic_error if the number of polygons or rings exceeds 31.
 * @throw cuspatial::logic_error polygon has less than 1 ring.
 * @throw cuspatial::logic_error polygon has less than 4 vertices.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OutputIt>
OutputIt point_in_polygon(Cart2dItA test_points_first,
                          Cart2dItA test_points_last,
                          OffsetIteratorA polygon_offsets_first,
                          OffsetIteratorA polygon_offsets_last,
                          OffsetIteratorB poly_ring_offsets_first,
                          OffsetIteratorB poly_ring_offsets_last,
                          Cart2dItB polygon_points_first,
                          Cart2dItB polygon_points_last,
                          OutputIt output,
                          rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_in_polygon.cuh>
