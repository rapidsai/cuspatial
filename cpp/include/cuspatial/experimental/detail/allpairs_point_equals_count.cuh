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
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {

namespace detail {

template <typename T>
struct allpairs_point_equals_count_functor {
  vec_2d<T> __device__ operator()(vec_2d<T> loc) {}
};

template <typename T>
struct allpairs_point_equals_count_functor {
}

}  // namespace detail

template <class InputIt, class OutputIt, class T>
OutputIt allpairs_point_equals_count(InputIt lhs_first,
                                     InputIt rhs_first,
                                     InputIt lhs_last,
                                     InputIt rhs_last,
                                     OutputIt output,
                                     rmm::cuda_stream_view stream)
{
  static_assert(is_same_floating_point<T, iterator_vec_base_type<InputIt>>(),
                "Origin and input must have the same base floating point type.");

  return thrust::transform(rmm::exec_policy(stream),
                           lhs_first,
                           rhs_first,
                           lhs_last,
                           rhs_last,
                           output,
                           detail::allpairs_point_equals_count_functor<T>{});
}

}  // namespace cuspatial
