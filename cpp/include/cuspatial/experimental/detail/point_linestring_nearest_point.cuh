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

#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/detail/utility/linestring.cuh>
#include <cuspatial/detail/utility/traits.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>

#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief The kernel to compute point to linestring distance
 */
template <typename Cart2dItA,
          typename Cart2dItB,
          typename OffsetIterator,
          typename OutputIterator>
void __global__
pairwise_point_linestring_nearest_point_kernel(Cart2dItA points_first,
                                               OffsetIterator linestring_offsets_first,
                                               OffsetIterator linestring_offsets_last,
                                               Cart2dItB linestring_points_first,
                                               Cart2dItB linestring_points_last,
                                               OutputIterator nearest_points_linestring_idx_zipped)
{
  using T = iterator_vec_base_type<Cart2dItA>;

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < std::distance(linestring_points_first, linestring_points_last);
       idx += gridDim.x * blockDim.x) {
    auto offsets_iter =
      thrust::upper_bound(thrust::seq, linestring_offsets_first, linestring_offsets_last, idx);
    // Pointer to the last point in the linestring.
    if (*offsets_iter - 1 == idx) { return; }
  }
}

}  // namespace detail

template <class Cart2dItA, class Cart2dItB, class OffsetIterator, class OutputItA, class OutputItB>
void pairwise_point_linestring_nearest_point(Cart2dItA points_first,
                                             Cart2dItA points_last,
                                             OffsetIterator linestring_offsets_first,
                                             Cart2dItB linestring_points_first,
                                             Cart2dItB linestring_points_last,
                                             OutputItA nearest_points,
                                             OutputItB nearest_point_linestring_idx,
                                             rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;
}

}  // namespace cuspatial
