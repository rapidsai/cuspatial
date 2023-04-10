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
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/algorithm/is_point_in_polygon.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cuspatial {

template <class MultiPolygonRangeA, class MultiPolygonRangeB, class OutputIt>
OutputIt pairwise_polygon_distance(MultiPolygonRangeA lhs,
                                   MultiPolygonRangeB rhs,
                                   OutputIt distances_first,
                                   rmm::cuda_stream_view stream)
{
  using T       = typename MultipolygonRangeA::element_t;
  using index_t = typename MultipolygonRangeB::index_t;

  CUSPATIAL_EXPECTS(lhs.size() == rhs.size(), "Must have the same number of input rows.");

  if (lhs.size() == 0) return distances_first;

  auto lhs_as_multipoints = lhs.as_multipoint_range();
  auto rhs_as_multipoints = rhs.as_multipoint_range();

  auto intersects = [&lhs, &rhs, &stream]() {
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

  std::size_t constexpr threads_per_block = 256;
  std::size_t const num_blocks = (lhs.num_points() + threads_per_block - 1) / threads_per_block;

  detail::linestring_distance<<<num_blocks, threads_per_block, 0, stream.value()>>>(
    lhs_as_multilinestrings, rhs_as_multilinestrings, intersects, distances_first);

  CUSPATIAL_CUDA_TRY(cudaGetLastError());
  return distances_first + lhs.size();
}

}  // namespace cuspatial
