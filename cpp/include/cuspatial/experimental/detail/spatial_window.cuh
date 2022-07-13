/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cuspatial/detail/utility/traits.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <type_traits>

namespace cuspatial {
namespace detail {

// Functor to filter out points that are not inside the query window
// This is passed to thrust::copy_if
template <typename T>
struct spatial_window_filter {
  spatial_window_filter(vec_2d<T> window_min, vec_2d<T> window_max)
    : min{std::min(window_min.x, window_max.x),   // support mirrored rectangles
          std::min(window_min.y, window_max.y)},  // where specified min > max
      max{std::max(window_min.x, window_max.x), std::max(window_min.y, window_max.y)}
  {
  }

  __device__ inline bool operator()(vec_2d<T> point)
  {
    return point.x > min.x && point.x < max.x && point.y > min.y && point.y < max.y;
  }

 protected:
  vec_2d<T> min;
  vec_2d<T> max;
};

}  // namespace detail

template <class InputIt, class T>
typename thrust::iterator_traits<InputIt>::difference_type count_points_in_spatial_window(
  vec_2d<T> window_min,
  vec_2d<T> window_max,
  InputIt points_first,
  InputIt points_last,
  rmm::cuda_stream_view stream)
{
  using Point       = typename std::iterator_traits<InputIt>::value_type;
  using OutputPoint = typename std::iterator_traits<InputIt>::value_type;

  static_assert(detail::is_convertible_to<cuspatial::vec_2d<T>, Point, OutputPoint>(),
                "Input and Output points must be convertible to cuspatial::vec_2d");

  static_assert(detail::is_same_floating_point<T,
                                               typename Point::value_type,
                                               typename OutputPoint::value_type>(),
                "Inputs and output must have the same value type.");

  return thrust::count_if(rmm::exec_policy(stream),
                          points_first,
                          points_last,
                          detail::spatial_window_filter{window_min, window_max});
}

template <class InputIt, class OutputIt, class T>
OutputIt points_in_spatial_window(vec_2d<T> window_min,
                                  vec_2d<T> window_max,
                                  InputIt points_first,
                                  InputIt points_last,
                                  OutputIt output_points_first,
                                  rmm::cuda_stream_view stream)
{
  using Point       = typename std::iterator_traits<InputIt>::value_type;
  using OutputPoint = typename std::iterator_traits<InputIt>::value_type;

  static_assert(detail::is_convertible_to<cuspatial::vec_2d<T>, Point, OutputPoint>(),
                "Input and Output points must be convertible to cuspatial::vec_2d");

  static_assert(detail::is_same_floating_point<T,
                                               typename Point::value_type,
                                               typename OutputPoint::value_type>(),
                "Inputs and output must have the same value type.");

  return thrust::copy_if(rmm::exec_policy(stream),
                         points_first,
                         points_last,
                         output_points_first,
                         detail::spatial_window_filter{window_min, window_max});
}

}  // namespace cuspatial
