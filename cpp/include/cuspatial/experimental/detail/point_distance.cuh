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

#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <type_traits>

namespace cuspatial {

template <class MultiPointArrayViewA, class MultiPointArrayViewB, class OutputIt>
OutputIt pairwise_point_distance(MultiPointArrayViewA multipoints1,
                                 MultiPointArrayViewB multipoints2,
                                 OutputIt distances_first,
                                 rmm::cuda_stream_view stream)
{
  using T = iterator_vec_base_type<typename MultiPointArrayViewA::point_it_t>;

  static_assert(
    is_same_floating_point<T, iterator_vec_base_type<typename MultiPointArrayViewB::point_it_t>>(),
    "Inputs must have the same floating point value type.");

  static_assert(is_same<vec_2d<T>,
                        typename MultiPointArrayViewA::point_t,
                        typename MultiPointArrayViewB::point_t>(),
                "All Input types must be cuspatial::vec_2d with the same value type");

  CUSPATIAL_EXPECTS(multipoints1.size() == multipoints2.size(),
                    "Inputs should have the same number of multipoints.");

  return thrust::transform(rmm::exec_policy(stream),
                           multipoints1.multipoint_begin(),
                           multipoints1.multipoint_end(),
                           multipoints2.multipoint_begin(),
                           distances_first,
                           [] __device__(auto& mp1, auto& mp2) {
                             T min_distance_squared = std::numeric_limits<T>::max();
                             for (vec_2d<T> const& p1 : mp1) {
                               for (vec_2d<T> const& p2 : mp2) {
                                 auto v               = p1 - p2;
                                 min_distance_squared = min(min_distance_squared, dot(v, v));
                               }
                             }
                             return sqrt(min_distance_squared);
                           });
}

}  // namespace cuspatial
