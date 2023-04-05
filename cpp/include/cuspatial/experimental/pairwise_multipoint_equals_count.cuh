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

#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <iterator>

namespace cuspatial {

/**
 * @brief Compute the number of multipoint pairs that are equal.
 *
 * Given two sets of multipoints, each represented by a range of `vec_2d<T>`s,
 * computes the number of pairs of multipoints that are equal. Example:
 *
 * ```
 * lhs: { {0, 0}, {1, 1}, {2, 2} }
 * rhs: { {0, 0}, {1, 1}, {2, 2} }
 * count: { 1, 1, 1 }
 *
 * lhs: { {0, 0} }
 * rhs: { {0, 0}, {1, 1}, {2, 2}, {3, 3} }
 * count: { 1 }
 *
 * lhs: { {0, 0}, {1, 1}, {2, 2}, {3, 3} }
 * rhs: { {0, 0} }
 * count: { 1, 0, 0, 0 }
 * ```
 *
 * @note All input iterators must have a `value_type` of `cuspatial::vec_2d<T>`
 * and the output iterator must be able to accept for storage values of type
 * `uint32_t`.
 *
 * @param[in]  lhs_first multipoint_ref of first set of points
 * @param[in]  rhs_first multipoint_ref of second set of points
 * @param[out] count_first: beginning of range of uint32_t counts
 * @param[in]  stream: The CUDA stream on which to perform computations and allocate memory.
 *
 * @tparam MultiPointRefA Iterator over multipoint vec_2ds. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam MultiPointRefB Iterator over multipoint vec_2ds. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Iterator over uint32_t. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 *
 * @return Output iterator to the element past the last count result written.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class MultiPointRangeA, class MultiPointRangeB, class OutputIt>
OutputIt pairwise_multipoint_equals_count(MultiPointRangeA lhs_first,
                                          MultiPointRangeB rhs_first,
                                          OutputIt count_first,
                                          rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/pairwise_multipoint_equals_count.cuh>
