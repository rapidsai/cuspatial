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
#include <cuspatial/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/detail/kernel/pairwise_distance.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <limits>

namespace cuspatial {

template <class MultiLinestringRange, class MultiPolygonRange, class OutputIt>
OutputIt pairwise_linestring_polygon_distance(MultiLinestringRange multilinestrings,
                                              MultiPolygonRange multipolygons,
                                              OutputIt distances_first,
                                              rmm::cuda_stream_view stream)
{
  using T       = typename MultiLinestringRange::element_t;
  using index_t = iterator_value_type<typename MultiLinestringRange::geometry_it_t>;

  CUSPATIAL_EXPECTS(multilinestrings.size() == multipolygons.size(),
                    "Must have the same number of input rows.");

  auto size = multilinestrings.size();

  if (size == 0) return distances_first;

  // Create a multipoint range from multilinestrings, computes intersection
  auto multipoints = multilinestrings.as_multipoint_range();
  auto intersects  = point_polygon_intersects(multipoints, multipolygons, stream);

  auto polygons_as_linestrings = multipolygons.as_multilinestring_range();

  thrust::transform(rmm::exec_policy(stream),
                    multilinestrings.begin(),
                    multilinestrings.end(),
                    multipolygons.begin(),
                    distances_first,
                    [] __device__(auto multilinestring, auto multipolygon) {
                      return (multilinestring.is_empty() || multipolygon.is_empty())
                               ? std::numeric_limits<T>::quiet_NaN()
                               : std::numeric_limits<T>::max();
                    });

  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings.num_points());

  detail::linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multilinestrings, polygons_as_linestrings, intersects.begin(), distances_first);

  return distances_first + size;
}

}  // namespace cuspatial
