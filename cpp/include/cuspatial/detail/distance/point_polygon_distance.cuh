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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>
#include <type_traits>

namespace cuspatial {

template <class MultiPointRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_point_polygon_distance(MultiPointRange multipoints,
                                         MultiPolygonRange multipolygons,
                                         OutputIt distances,
                                         rmm::cuda_stream_view stream)
{
  using T       = typename MultiPointRange::element_t;
  using index_t = typename MultiPointRange::index_t;

  CUSPATIAL_EXPECTS(multipoints.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  if (multipoints.size() == 0) return distances;

  auto intersects = point_polygon_intersects(multipoints, multipolygons, stream);

  auto polygons_as_linestrings = multipolygons.as_multilinestring_range();

  thrust::fill(rmm::exec_policy(stream),
               distances,
               distances + multipoints.size(),
               std::numeric_limits<T>::max());

  auto [threads_per_block, n_blocks] = grid_1d(polygons_as_linestrings.num_points());

  detail::point_linestring_distance<<<n_blocks, threads_per_block, 0, stream.value()>>>(
    multipoints, polygons_as_linestrings, intersects.begin(), distances);

  CUSPATIAL_CHECK_CUDA(stream.value());

  return distances + multipoints.size();
}

}  // namespace cuspatial
