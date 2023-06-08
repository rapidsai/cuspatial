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
 * @addtogroup spatial_relationship
 * @{
 */

/**
 * @brief Tests whether the specified points are inside any of the specified polygons.
 *
 * Tests whether points are inside at most 31 polygons. Polygons are a collection of one or more
 * rings. Rings are a collection of three or more vertices.
 *
 * Each input point will map to one `int32_t` element in the output. Each bit (except the sign bit)
 * represents a hit or miss for each of the input polygons in least-significant-bit order. i.e.
 * `output[3] & 0b0010` indicates a hit or miss for the 3rd point against the 2nd polygon.
 *
 * Note that the input must be a single geometry column, that is a (multi*)geometry_range
 * initialized with counting iterator as the geometry offsets iterator.
 *
 * @tparam PointRange an instance of template type `multipoint_range`, where
 * `GeometryIterator` must be a counting iterator
 * @tparam PolygonRange an instance of template type `multipolygon_range`, where
 * `GeometryIterator` must be a counting iterator
 * @tparam OutputIt iterator type for output array. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI], be device-accessible, mutable and iterate on `int32_t`
 * type.
 *
 * @param points Range of points, one per computed point-in-polygon pair,
 * @param polygons Range of polygons, one per comptued point-in-polygon pair
 * @param output begin iterator to the output buffer
 * @param stream The CUDA stream to use for kernel launches.
 * @return iterator to one past the last element in the output buffer
 *
 * @note Direction of rings does not matter.
 * @note The points of the rings must be explicitly closed.
 * @note Overlapping rings negate each other. This behavior is not limited to a single negation,
 * allowing for "islands" within the same polygon.
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
 * @pre Output iterator must be mutable and iterate on int32_t type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class PointRange, class PolygonRange, class OutputIt>
OutputIt point_in_polygon(PointRange points,
                          PolygonRange polygons,
                          OutputIt output,
                          rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Given (point, polygon) pairs, tests whether the point in the pair is in the polygon in the
 * pair.
 *
 * Note that the input must be a single geometry column, that is a (multi*)geometry_range
 * initialized with counting iterator as the geometry offsets iterator.
 *
 * Each input point will map to one `uint8_t` element in the output.
 *
 * @tparam PointRange an instance of template type `multipoint_range`, where
 * `GeometryIterator` must be a counting iterator
 * @tparam PolygonRange an instance of template type `multipolygon_range`, where
 * `GeometryIterator` must be a counting iterator
 * @tparam OutputIt iterator type for output array. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI], be device-accessible, mutable and iterate on `int32_t`
 * type.
 *
 * @param points Range of points, one per computed point-in-polygon pair,
 * @param polygons Range of polygons, one per comptued point-in-polygon pair
 * @param output begin iterator to the output buffer
 * @param stream The CUDA stream to use for kernel launches.
 * @return iterator to one past the last element in the output buffer
 *
 * @note Direction of rings does not matter.
 * @note The points of the rings must be explicitly closed.
 * @note Overlapping rings negate each other. This behavior is not limited to a single negation,
 * allowing for "islands" within the same polygon.
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
 * @pre Output iterator must be mutable and iterate on uint8_t type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <typename PointRange, typename PolygonRange, typename OutputIt>
OutputIt pairwise_point_in_polygon(PointRange points,
                                   PolygonRange polygons,
                                   OutputIt results,
                                   rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @} // end of doxygen group
 */

}  // namespace cuspatial

#include <cuspatial/detail/point_in_polygon.cuh>
