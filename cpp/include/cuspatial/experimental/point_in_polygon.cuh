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
 * @copybrief
 */
template <class Cart2dItA, class Cart2dItB, class OffsetIteratorA, class OffsetIteratorB , class OutputIt>
OutputIt point_in_polygon(Cart2dItA points_begin,
                          Cart2dItA points_end,
                          OffsetIteratorA polygon_offsets_begin,
                          OffsetIteratorA polygon_offsets_end,
                          OffsetIteratorB ring_offsets_begin,
                          OffsetIteratorB ring_offsets_end,
                          Cart2dItB polygon_points_begin,
                          Cart2dItB polygon_points_end,
                          OutputIt output,
                          rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_in_polygon.cuh>
