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

#include <cuspatial/detail/utility/traits.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <type_traits>

namespace cuspatial {

template <class Cart2dItA, class Cart2dItB, class OutputIt>
OutputIt pairwise_point_distance(Cart2dItA points1_first,
                                 Cart2dItA points1_last,
                                 Cart2dItB points2_first,
                                 OutputIt distances_first,
                                 rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  static_assert(
    detail::is_floating_point<T,
                              typename std::iterator_traits<Cart2dItB>::value_type::value_type,
                              typename std::iterator_traits<OutputIt>::value_type>(),
    "Inputs and output must be floating point types.");

  static_assert(detail::is_same<T, typename std::iterator_traits<OutputIt>::value_type>(),
                "Inputs and output must be the same types.");

  static_assert(detail::is_same<cartesian_2d<T>,
                                typename std::iterator_traits<Cart2dItA>::value_type,
                                typename std::iterator_traits<Cart2dItB>::value_type>(),
                "All Input types must be cuspatial::cartesian_2d with the same value type");

  return thrust::transform(rmm::exec_policy(stream),
                           points1_first,
                           points1_last,
                           points2_first,
                           distances_first,
                           [] __device__(auto p1, auto p2) {
                             auto v = p1 - p2;
                             return std::sqrt(dot(v, v));
                           });
}

}  // namespace cuspatial
