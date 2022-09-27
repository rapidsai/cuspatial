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

#include <cuspatial/constants.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <type_traits>

namespace cuspatial {

namespace detail {

template <typename T>
struct haversine_distance_functor {
  haversine_distance_functor(T radius) : radius_(radius) {}

  __device__ T operator()(vec_2d<T> lonlat_a, vec_2d<T> lonlat_b)
  {
    auto ax = lonlat_a.x * DEGREE_TO_RADIAN;
    auto ay = lonlat_a.y * DEGREE_TO_RADIAN;
    auto bx = lonlat_b.x * DEGREE_TO_RADIAN;
    auto by = lonlat_b.y * DEGREE_TO_RADIAN;

    // haversine formula
    auto x        = (bx - ax) / 2;
    auto y        = (by - ay) / 2;
    auto sinysqrd = sin(y) * sin(y);
    auto sinxsqrd = sin(x) * sin(x);
    auto scale    = cos(ay) * cos(by);

    return 2 * radius_ * asin(sqrt(sinysqrd + sinxsqrd * scale));
  }

  T radius_{};
};

}  // namespace detail

template <class LonLatItA, class LonLatItB, class OutputIt, class T>
OutputIt haversine_distance(LonLatItA a_lonlat_first,
                            LonLatItA a_lonlat_last,
                            LonLatItB b_lonlat_first,
                            OutputIt distance_first,
                            T const radius,
                            rmm::cuda_stream_view stream)
{
  static_assert(
    is_same<vec_2d<T>, iterator_value_type<LonLatItA>, iterator_value_type<LonLatItB>>(),
    "Inputs must be cuspatial::vec_2d");
  static_assert(
    is_same_floating_point<T,
                           typename cuspatial::iterator_vec_base_type<LonLatItA>,
                           typename cuspatial::iterator_value_type<OutputIt>>(),
    "All iterator types and radius must have the same floating-point coordinate value type.");

  CUSPATIAL_EXPECTS(radius > 0, "radius must be positive.");

  return thrust::transform(rmm::exec_policy(stream),
                           a_lonlat_first,
                           a_lonlat_last,
                           b_lonlat_first,
                           distance_first,
                           detail::haversine_distance_functor<T>(radius));
}

}  // namespace cuspatial
