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
 * @brief Compute pairwise multipoint to multilinestring distance
 *
 * @tparam MultiPointArrayView an instance of template type `multipoint_range`
 * @tparam MultiLinestringArrayView an instance of template type `multilinestring_range`
 *
 * @param multipoints Range object of a multipoint array
 * @param multilinestrings Range object of a multilinestring array
 * @param stream The CUDA stream to use for device memory operations and kernel launches.
 * @return Output iterator to the element past the last distance computed.
 */
template <class MultiPointArrayView, class MultiLinestringArrayView, class OutputIt>
OutputIt pairwise_point_linestring_distance(
  MultiPointArrayView multipoints,
  MultiLinestringArrayView multilinestrings,
  OutputIt distances_first,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/point_linestring_distance.cuh>
