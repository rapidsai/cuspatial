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

template <class Cart2dItA, class Cart2dItB, class OffsetIterator, class OutputItA, class OutputItB>
void pairwise_point_linestring_nearest_point(
  Cart2dItA points_first,
  Cart2dItA points_last,
  OffsetIterator linestring_offsets_first,
  Cart2dItB linestring_points_first,
  Cart2dItB linestring_points_last,
  OutputItA nearest_points,
  OutputItB nearest_point_linestring_idx,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial
