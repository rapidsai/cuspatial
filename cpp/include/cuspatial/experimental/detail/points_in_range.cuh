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

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/iterator_traits.h>

#include <type_traits>

namespace cuspatial {
namespace detail {

// Functor to filter out points that are not inside the query range. Passed to thrust::copy_if
template <typename T>
struct range_filter {
  range_filter(vec_2d<T> vertex_1, vec_2d<T> vertex_2)
    : v1{std::min(vertex_1.x, vertex_2.x), std::min(vertex_1.y, vertex_2.y)},
      v2{std::max(vertex_1.x, vertex_2.x), std::max(vertex_1.y, vertex_2.y)}
  {
  }

  __device__ inline bool operator()(vec_2d<T> point)
  {
    return point.x > v1.x && point.x < v2.x && point.y > v1.y && point.y < v2.y;
  }

 protected:
  vec_2d<T> v1;
  vec_2d<T> v2;
};

}  // namespace detail

template <class InputIt, class T>
typename thrust::iterator_traits<InputIt>::difference_type count_points_in_range(
  vec_2d<T> vertex_1,
  vec_2d<T> vertex_2,
  InputIt points_first,
  InputIt points_last,
  rmm::cuda_stream_view stream)
{
  using Point = typename std::iterator_traits<InputIt>::value_type;

  static_assert(cuspatial::is_convertible_to<cuspatial::vec_2d<T>, Point>(),
                "Input points must be convertible to cuspatial::vec_2d");

  return thrust::count_if(
    rmm::exec_policy(stream), points_first, points_last, detail::range_filter{vertex_1, vertex_2});
}

template <class InputIt, class OutputIt, class T>
OutputIt copy_points_in_range(vec_2d<T> vertex_1,
                              vec_2d<T> vertex_2,
                              InputIt points_first,
                              InputIt points_last,
                              OutputIt output_points_first,
                              rmm::cuda_stream_view stream)
{
  using Point = typename std::iterator_traits<InputIt>::value_type;

  static_assert(cuspatial::is_convertible_to<cuspatial::vec_2d<T>, Point>(),
                "Input points must be convertible to cuspatial::vec_2d");

  static_assert(is_same_floating_point<T, typename Point::value_type>(),
                "Inputs and Range coordinates must have the same value type.");

  return thrust::copy_if(rmm::exec_policy(stream),
                         points_first,
                         points_last,
                         output_points_first,
                         detail::range_filter{vertex_1, vertex_2});
}

}  // namespace cuspatial
