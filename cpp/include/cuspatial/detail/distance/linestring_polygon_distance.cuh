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
#include "linestring_distance.cuh"

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/detail/functors.cuh>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/range.cuh>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/zip_function.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
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

  thrust::fill(rmm::exec_policy(stream),
               distances_first,
               distances_first + size.size(),
               std::numeric_limits<T>::max());

  auto [threads_per_block, num_blocks] = grid_1d(multilinestrings.num_points());

  detail::linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    multilinestrings, polygons_as_linestrings, intersects.begin(), distances_first);

  return distances_first + multilinestrings.num_multilinestrings();
}

}  // namespace cuspatial
