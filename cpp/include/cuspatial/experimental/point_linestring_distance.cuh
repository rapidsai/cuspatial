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
 * @ingroup distance
 * @copybrief cuspatial::pairwise_point_linestring_distance
 *
 * The number of distances computed is `std::distance(points_first, points_last)`.
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
 * @param distances_first beginning of the output range of distances
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 *
 * @pre all input iterators for coordinates must have `cuspatial::vec_2d` type.
 * @pre all scalar types must be floating point types, and must be the same type for all input
 * iterators and output iterators.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_distance(
  OffsetIteratorA point_geometry_offset_first,
  OffsetIteratorA point_geometry_offset_last,
  Cart2dItA points_first,
  Cart2dItA points_last,
  OffsetIteratorB linestring_geometry_offset_first,
  OffsetIteratorC linestring_part_offsets_first,
  OffsetIteratorC linestring_part_offsets_last,
  Cart2dItB linestring_points_first,
  Cart2dItB linestring_points_last,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_linestring_distance.cuh>
