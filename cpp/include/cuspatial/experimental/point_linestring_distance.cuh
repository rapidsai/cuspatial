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

#include <cuspatial/experimental/array_view/multilinestring_array.cuh>
#include <cuspatial/experimental/array_view/multipoint_array.cuh>

namespace cuspatial {

/**
 * @brief Compute pairwise multipoint to multilinestring distance
 *
 * @param multipoints Array view object of a multipoint array
 * @param multilinestrings Array view object of a multilinestring array
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @return Output iterator to the element past the last tuple computed.
 */
template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OffsetIteratorC,
          class OutputIt>
OutputIt pairwise_point_linestring_distance(
  array_view::multipoint_array<OffsetIteratorA, Cart2dItA> multipoints,
  array_view::multilinestring_array<OffsetIteratorB, OffsetIteratorC, Cart2dItB> multilinestrings,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_linestring_distance.cuh>
