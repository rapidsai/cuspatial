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
#include <cuspatial/experimental/iterator_collections.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <type_traits>

namespace cuspatial {

template <class OffsetIteratorA,
          class OffsetIteratorB,
          class Cart2dItA,
          class Cart2dItB,
          class OutputIt>
OutputIt pairwise_point_distance(
  iterator_collections::multipoint_array<OffsetIteratorA, Cart2dItA> multipoints1,
  iterator_collections::multipoint_array<OffsetIteratorB, Cart2dItB> multipoints2,
  OutputIt distances_first,
  rmm::cuda_stream_view stream)
{
  using T = typename detail::iterator_vec_base_type<Cart2dItA>;

  static_assert(detail::is_same_floating_point<T,
                                               typename detail::iterator_vec_base_type<Cart2dItB>,
                                               typename detail::iterator_value_type<OutputIt>>(),
                "Inputs and output must have the same floating point value type.");

  static_assert(detail::is_same<vec_2d<T>,
                                typename detail::iterator_value_type<Cart2dItA>,
                                typename detail::iterator_value_type<Cart2dItB>>(),
                "All Input types must be cuspatial::vec_2d with the same value type");

  return thrust::transform(
    rmm::exec_policy(stream),
    multipoints1.multipoint_begin(),
    multipoints1.multipoint_end(),
    multipoints2.multipoint_begin(),
    distances_first,
    [] __device__(auto& mp1, auto& mp2) {
      T min_distance_squared;
      thrust::for_each(thrust::seq,
                       mp1.point_begin(),
                       mp1.point_end(),
                       [mp2, &min_distance_squared] __device__(auto& p1) {
                         thrust::for_each(thrust::seq,
                                          mp2.point_begin(),
                                          mp2.point_end(),
                                          [&p1, &min_distance_squared] __device__(auto& p2) {
                                            auto v               = p1 - p2;
                                            auto d               = dot(v, v);
                                            min_distance_squared = min(min_distance_squared, d);
                                          });
                       });
      return sqrt(min_distance_squared);
    });
}

}  // namespace cuspatial
