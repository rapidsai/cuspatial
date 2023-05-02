/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
 * @brief Computes pairwise multipolygon to multipolygon distance
 *
 * @tparam MultiPolygonRangeA An instance of template type `multipolygon_range`
 * @tparam MultiPolygonRangeB An instance of template type `multipolygon_range`
 * @tparam OutputIt iterator type for output array. Must meet the requirements of [LRAI](LinkLRAI).
 * Must be an iterator to type convertible from floating points.
 *
 * @param lhs The first multipolygon range to compute distance from
 * @param rhs The second multipolygon range to compute distance to
 * @param stream The CUDA stream on which to perform computations
 * @return Output Iterator past the last distance computed
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultipolygonRangeA, class MultipolygonRangeB, class OutputIt>
OutputIt pairwise_polygon_distance(MultipolygonRangeA lhs,
                                   MultipolygonRangeB rhs,
                                   OutputIt distances_first,
                                   rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial

#include <cuspatial/detail/polygon_distance.cuh>
