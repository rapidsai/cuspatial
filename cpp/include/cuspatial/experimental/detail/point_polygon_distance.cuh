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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/memory.h>

#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @brief Kernel to compute the distance between pairs of point and polygon.
 */
template <class MultiPointRange, class MultiPolygonRange, class OutputIterator>
void __global__ pairwise_point_polygon_distance_kernel(MultiPointRange multipoints,
                                                       MultiPolygonRange multipolygons,
                                                       OutputIterator distances)
{
  using T = typename MultiPointRange::element_t;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < multipolygons.num_points();
       idx += gridDim.x * blockDim.x) {
    auto geometry_idx = multipolygons.geometry_idx_from_point_idx(idx);
  }
}

}  // namespace detail
template <class MultiPointRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_point_polygon_distance(MultiPointRange multipoints,
                                         MultiPolygonRange multipoiygons,
                                         OutputIt distances_first,
                                         rmm::cuda_stream_view stream)
{
  using T = typename MultiPointRange::element_t;

  static_assert(is_same_floating_point<T, typename MultiPolygonRange::element_t>(),
                "Inputs must have same floating point value type.");

  static_assert(
    is_same<vec_2d<T>, typename MultiPointRange::point_t, typename MultiPolygonRange::point_t>(),
    "Inputs must be cuspatial::vec_2d");

  CUSPATIAL_EXPECTS(multipoints.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  auto [threads_per_block, n_blocks] = grid_id(multipolygons.num_points());

  pairwise_point_polygon_distance_kernel<<<threads_per_block, n_blocks, 0, stream.value()>>>(
    multipoints, multipolygons, distances_first);

  CUSPATIAL_CHECK_CUDA(stream.value());
}

}  // namespace cuspatial
