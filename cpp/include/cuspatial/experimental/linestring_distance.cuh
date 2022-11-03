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
 * @copybrief cuspatial::pairwise_linestring_distance
 *
 * The shortest distance between two linestrings is defined as the shortest distance
 * between all pairs of segments of the two linestrings. If any of the segments intersect,
 * the distance is 0.
 *
 * @tparam MultiLinestringRange an instance of template type `multilinestring_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI)
 * and be device writable.
 *
 * @param multilinestrings1 Range object of the lhs multilinestring array
 * @param multilinestrings2 Range object of the rhs multilinestring array
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @return Output iterator to the element past the last distance computed.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultiLinestringRange1, class MultiLinstringRange2, class OutputIt>
OutputIt pairwise_linestring_distance(MultiLinestringRange1 multilinestrings1,
                                      MultiLinstringRange2 multilinestrings2,
                                      OutputIt distances_first,
                                      rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/linestring_distance.cuh>
