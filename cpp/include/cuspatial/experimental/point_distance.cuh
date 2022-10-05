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

#include <cuspatial/experimental/array_view/multipoint_array.cuh>

namespace cuspatial {

/**
 * @ingroup distance
 * @copybrief cuspatial::pairwise_point_distance
 *
 * @param multipoints1 Array view object constructed from the iterators of the first multipoint in
 * the pair.
 * @param multipoints2 Array view object constructed from the iterators of the second multipoint in
 * the pair.
 * @return Iterator past the last distance computed
 */
template <class OffsetIteratorA,
          class OffsetIteratorB,
          class Cart2dItA,
          class Cart2dItB,
          class OutputIt>
OutputIt pairwise_point_distance(
  array_view::multipoint_array<OffsetIteratorA, Cart2dItA> multipoints1,
  array_view::multipoint_array<OffsetIteratorB, Cart2dItB> multipoints2,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_distance.cuh>
