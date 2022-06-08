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
 * @copybrief cuspatial::pairwise_point_distance
 *
 * Computes cartesian distances between points.
 *
 * @tparam Cart2dItA iterator type for point array of the first point of each pair. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam Cart2dItB iterator type for point array of the second point of each pair. Must meet
 * the requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt iterator type for output array. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI], be mutable and be device-accessible.
 *
 * @param points1_first beginning of range of the point of the first point of each
 * pair
 * @param points1_last end of range of the point of the first point of each pair
 * @param points2_first beginning of range of the point of the second point of each
 * pair
 * @param distances_first beginning iterator to output
 * @param stream The CUDA stream to use for device memory operations and kernel launches
 * @return Output iterator to one past the last element in the output range
 *
 * @pre all input iterators for coordinates must have `cuspatial::cartesian_2d` type.
 * @pre all scalar types must be floating point types, and must be the same type for all input
 * iterators and output iterators.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class Cart2dItA, class Cart2dItB, class OutputIt>
OutputIt pairwise_point_distance(Cart2dItA points1_first,
                                 Cart2dItA points1_last,
                                 Cart2dItB points2_first,
                                 OutputIt distances_first,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_distance.cuh>
