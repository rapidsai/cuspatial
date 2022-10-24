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
 * @tparam MultiPointArrayViewA An instance of template type `array_view::multipoint_array`
 * @tparam MultiPointArrayViewB An instance of template type `array_view::multipoint_array`
 *
 * @param multipoints1 Range of first multipoint in each distance pair.
 * @param multipoints2 Range of second multipoint in each distance pair.
 * @return Iterator past the last distance computed
 */
template <class MultiPointArrayViewA, class MultiPointArrayViewB, class OutputIt>
OutputIt pairwise_point_distance(MultiPointArrayViewA multipoints1,
                                 MultiPointArrayViewB multipoints2,
                                 OutputIt distances_first,
                                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);
}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_distance.cuh>
