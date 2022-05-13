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
 * @copybrief cuspatial::pairwise_linestring_distance
 *
 * The shortest distance between two linestrings is defined as the shortest distance
 * between all pairs of segments of the two linestrings. If any of the segments intersect,
 * the distance is 0.
 *
 * @tparam Cart2dItA iterator type for point array of the first linestring of each pair. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Cart2dItB iterator type for point array of the second linestring of each pair. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OffsetIterator iterator type for offset array. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt iterator type for output array. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam std::iterator_traits<Cart2dItA>::value_type value type of `Cart2dItA`, must be
 * `cuspatial::cartesian_2d`
 * @tparam std::iterator_traits<Cart2dItB>::value_type value type of `Cart2dItB`, must be
 * `cuspatial::cartesian_2d`
 *
 * @param linestring1_offsets_first begin of range of the offsets to the first linestring of each
 * pair
 * @param linestring1_offsets_last end of range of the offsets to the first linestring of each pair
 * @param linestring1_points_first begin of range of the point of the first linestring of each pair
 * @param linestring1_points_last end of range of the point of the first linestring of each pair
 * @param linestring2_offsets_first begin of range of the offsets to the second linestring of each
 * pair
 * @param linestring2_points_first begin of range of the point of the second linestring of each pair
 * @param linestring2_points_last end of range of the point of the second linestring of each pair
 * @param distances_first begin to output array
 * @param stream Used for device memory operations and kernel launches.
 *
 * @pre all input iterators for coordinates must have `cuspatial::cartesian_2d` type.
 * @pre all scalar types must be floating point types, and must be the same type for all input
 * iterators and output iterators.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIterator,
          class OutputIt,
          class Cart2dA = typename std::iterator_traits<Cart2dItA>::value_type,
          class Cart2dB = typename std::iterator_traits<Cart2dItB>::value_type>
void pairwise_linestring_distance(OffsetIterator linestring1_offsets_first,
                                  OffsetIterator linestring1_offsets_last,
                                  Cart2dItA linestring1_points_first,
                                  Cart2dItA linestring1_points_last,
                                  OffsetIterator linestring2_offsets_first,
                                  Cart2dItB linestring2_points_first,
                                  Cart2dItB linestring2_points_last,
                                  OutputIt distances_first,
                                  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/linestring_distance.cuh>
