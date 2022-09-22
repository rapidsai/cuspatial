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
 * @brief Compute the nearest points and geometry ID between pairs of multipoint and
 * multilinestring
 *
 * The nearest point from a test point to a linestring is a point on the linestring that has
 * the shortest distance to the test point compared to any other points on the linestring.
 *
 * The nearest point from a test multipoint to a multilinestring is the nearest point that
 * has the shortest distance in all pairs of points and linestrings.
 *
 * In addition, this API writes these geometry and part ID where the nearest point locates to output
 * iterators:
 * - The point ID indicates which point in the multipoint is the nearest point.
 * - The linestring ID is the offset within the multilinestring that contains the nearest point.
 * - The segment ID is the offset within the linestring of the segment that contains the nearest
 * point. It is the same as the ID of the starting point of the segment.
 *
 * @tparam Cart2dItA iterator type for point array of the point element of each pair. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Cart2dItB iterator type for point array of the linestring element of each pair. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorA iterator type for `point_geometry_offset` array. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorB iterator type for `linestring_geometry_offset` array. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIteratorC iterator type for `linestring_part_offset` array. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt iterator type for output array. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 *
 * @param point_geometry_offset_first beginning of the range of multipoint geometries of each
 * pair
 * @param point_geometry_offset_last end of the range of multipoint geometries of each pair
 * @param points_first beginning of the range of point values
 * @param points_last end of the range of the point values
 * @param linestring_geometry_offset_first beginning of the range of offsets to the multilinestring
 * geometry of each pair, the end range is implied by linestring_geometry_offset_first +
 * std::distance(`point_geometry_offset_first`, `point_geometry_offset_last`)
 * @param linestring_offsets_first beginning of the range of offsets to the starting point
 * of each linestring
 * @param linestring_offsets_last end of the range of offsets to the starting point
 * of each linestring
 * @param linestring_points_first beginning of the range of linestring points
 * @param linestring_points_last end of the range of linestring points
 * @param output_first A zipped-iterator of 4 outputs. The first element should be compatible
 * with iterator_value_type<OffsetIteratorA>, stores the geometry ID of the nearest point in
 * multipoint. The second element should be compatible with iterator_value_type<OffsetIteratorB>,
 * stores the geometry ID of the nearest linestring. The third element should be compatible with
 * iterator_value_type<OffsetIteratorC>, stores the part ID to the nearest segment. The forth
 * element should be compatible with vec_2d, stores the coordinate of the nearest point on the
 * (multi)linestring.
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @return Output iterator to the element past the last tuple computed.
 *
 * @pre all input iterators for coordinates must have `cuspatial::vec_2d` type,
 * and must have the same base floating point type.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class Vec2dItA,
          class Vec2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_nearest_points(
  OffsetIteratorA points_geometry_offsets_first,
  OffsetIteratorA points_geometry_offsets_last,
  Vec2dItA points_first,
  Vec2dItA points_last,
  OffsetIteratorB linestring_geometry_offsets_first,
  OffsetIteratorC linestring_part_offsets_first,
  OffsetIteratorC linestring_part_offsets_last,
  Vec2dItB linestring_points_first,
  Vec2dItB linestring_points_last,
  OutputIt output_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_linestring_nearest_points.cuh>
