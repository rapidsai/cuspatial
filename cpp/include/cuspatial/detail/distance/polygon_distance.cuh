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

#include "distance_utils.cuh"

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/kernel/pairwise_distance.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>
#include <thrust/transform.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuspatial {

/**
 * @brief Implementation of pairwise distance between two multipolygon ranges.
 *
 * All points in lhs and rhs are tested for intersection its corresponding pair,
 * and if any intersection is found, the distance between the two polygons is 0.
 * Otherwise, the distance is the minimum distance between any two segments in the
 * multipolygon pair.
 */
template <class MultipolygonRangeA, class MultipolygonRangeB, class OutputIt>
OutputIt pairwise_polygon_distance(MultipolygonRangeA lhs,
                                   MultipolygonRangeB rhs,
                                   OutputIt distances_first,
                                   rmm::cuda_stream_view stream)
{
  using T = typename MultipolygonRangeA::element_t;

  CUSPATIAL_EXPECTS(lhs.size() == rhs.size(), "Must have the same number of input rows.");

  if (lhs.size() == 0) return distances_first;

  auto lhs_as_multipoints = lhs.as_multipoint_range();
  auto rhs_as_multipoints = rhs.as_multipoint_range();

  auto intersects = [&]() {
    auto lhs_in_rhs = point_polygon_intersects(lhs_as_multipoints, rhs, stream);
    auto rhs_in_lhs = point_polygon_intersects(rhs_as_multipoints, lhs, stream);

    rmm::device_uvector<uint8_t> intersects(lhs_in_rhs.size(), stream);
    thrust::transform(rmm::exec_policy(stream),
                      lhs_in_rhs.begin(),
                      lhs_in_rhs.end(),
                      rhs_in_lhs.begin(),
                      intersects.begin(),
                      thrust::logical_or<uint8_t>{});
    return intersects;
  }();

  auto lhs_as_multilinestrings = lhs.as_multilinestring_range();
  auto rhs_as_multilinestrings = rhs.as_multilinestring_range();

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + lhs.size(),
               std::numeric_limits<T>::max());

  auto [threads_per_block, num_blocks] = grid_1d(lhs.num_points());

  detail::linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    lhs_as_multilinestrings, rhs_as_multilinestrings, intersects.begin(), distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
  return distances_first + lhs.size();
}

}  // namespace cuspatial
