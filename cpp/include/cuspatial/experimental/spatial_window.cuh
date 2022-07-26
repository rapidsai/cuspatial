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

#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/iterator_traits.h>

namespace cuspatial {

/**
 * @brief Count of points (x,y) that fall within a rectangular query window.
 *
 * @ingroup spatial_relationship
 *
 * A point (x, y) is in the window if `x > window_min_x && x < window_max_x && y > window_min_y &&
 * y < window_max_y`.
 *
 * Swaps `window_min.x` and `window_max.x` if `window_min.x > window_max.x`.
 * Swaps `window_min.y` and `window_max.y` if `window_min.y > window_max.y`.
 *
 * @param[in] window_min lower-left (x, y) coordinate of the query window
 * @param[in] window_max upper-right (x, y) coordinate of the query window
 * @param[in] points_first beginning of range of (x, y) coordinates of points to be queried
 * @param[in] points_last end of range of (x, y) coordinates of points to be queried
 * @param[in] stream The CUDA stream on which to perform computations
 *
 * @tparam InputIt Iterator to input points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam T The underlying coordinate type. Must be a floating-point type.
 *
 * @pre All iterators must have the same `value_type`, with the same underlying floating-point
 * coordinate type (e.g. `cuspatial::vec_2d<float>`).
 *
 * @returns The number of input points that fall within the specified window.
 */
template <class InputIt, class T>
typename thrust::iterator_traits<InputIt>::difference_type count_points_in_spatial_window(
  vec_2d<T> window_min,
  vec_2d<T> window_max,
  InputIt points_first,
  InputIt points_last,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Find all points (x,y) that fall within a rectangular query window.
 *
 * @ingroup spatial_relationship
 *
 * A point (x, y) is in the window if `x > window_min_x && x < window_max_x && y > window_min_y &&
 * y < window_max_y`.
 *
 * Swaps `window_min.x` and `window_max.x` if `window_min.x > window_max.x`.
 * Swaps `window_min.y` and `window_max.y` if `window_min.y > window_max.y`.
 *
 * @param[in] window_min lower-left (x, y) coordinate of the query window
 * @param[in] window_max upper-right (x, y) coordinate of the query window
 * @param[in] points_first beginning of range of (x, y) coordinates of points to be queried
 * @param[in] points_last end of range of (x, y) coordinates of points to be queried
 * @param[out] output_points_first beginning of output range of (x, y) coordinates within the
 * query window
 * @param[in]  stream: The CUDA stream on which to perform computations and allocate memory.
 *
 * @tparam InputIt Iterator to input points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Output iterator. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 * @tparam T The underlying coordinate type. Must be a floating-point type.
 *
 * @pre The range `[points_first, points_last)` may equal the range `[output_points_first,
 * output_points_first + std::distance(points_first, points_last)), but the ranges may not
 * partially overlap.
 * @pre All iterators must have the same `value_type`, with the same underlying floating-point
 * coordinate type (e.g. `cuspatial::vec_2d<float>`).
 *
 * @returns Output iterator to the element past the last output point.
 */
template <class InputIt, class OutputIt, class T>
OutputIt points_in_spatial_window(vec_2d<T> window_min,
                                  vec_2d<T> window_max,
                                  InputIt points_first,
                                  InputIt points_last,
                                  OutputIt output_points_first,
                                  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/spatial_window.cuh>
