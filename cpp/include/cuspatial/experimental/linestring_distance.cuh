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
                                  rmm::cuda_stream_view stream);

}

#include <cuspatial/experimental/detail/linestring_distance.cuh>
