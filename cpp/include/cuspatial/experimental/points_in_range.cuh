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
 * @brief Count of points (x,y) that fall within a query range.
 *
 * @ingroup spatial_relationship
 *
 * The query range is defined by a pair of opposite vertices within the coordinate system of the
 * input points, `v1` and `v2`. A point (x, y) is in the range if `x` lies between `v1.x` and `v2.x`
 * and `y` lies between `v1.y` and `v2.y`. A point is only counted if it is strictly within the
 * interior of the query range. Points exactly on an edge or vertex of the range are not counted.
 *
 * The query range vertices and the input points are assumed to be defined in the same coordinate
 * system.
 *
 * @param[in] vertex_1 Vertex of the query range quadrilateral
 * @param[in] vertex_2 Vertex of the query range quadrilateral opposite `vertex_1`
 * @param[in] points_first beginning of sequence of (x, y) coordinates of points to be queried
 * @param[in] points_last end of sequence of (x, y) coordinates of points to be queried
 * @param[in] stream The CUDA stream on which to perform computations
 *
 * @tparam InputIt Iterator to input points. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam T The underlying coordinate type. Must be a floating-point type.
 *
 * @pre All iterators must have the same `value_type`, with the same underlying floating-point
 * coordinate type (e.g. `cuspatial::vec_2d<float>`).
 *
 * @returns The number of input points that fall within the specified query range.
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class InputIt, class T>
typename thrust::iterator_traits<InputIt>::difference_type count_points_in_range(
  vec_2d<T> vertex_1,
  vec_2d<T> vertex_2,
  InputIt points_first,
  InputIt points_last,
  rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Copies points (x,y) that fall within a query range.
 *
 * @ingroup spatial_relationship
 *
 * The query range is defined by a pair of opposite vertices of a quadrilateral within the
 * coordinate system of the input points, `v1` and `v2`. A point (x, y) is in the range if `x` lies
 * between `v1.x` and `v2.x` and `y` lies between `v1.y` and `v2.y`. A point is only counted if it
 * is strictly within the interior of the query range. Points exactly on an edge or vertex of the
 * range are not copied.
 *
 * The query range vertices and the input points are assumed to be defined in the same coordinate
 * system.
 *
 * `output_points_first` must be an iterator to storage of sufficient size for the points that will
 * be copied. cuspatial::count_points_in_range may be used to determine the size required.
 *
 * @param[in] vertex_1 Vertex of the query range quadrilateral
 * @param[in] vertex_2 Vertex of the query range quadrilateral opposite `vertex_1`
 * @param[in] points_first beginning of sequence of (x, y) coordinates of points to be queried
 * @param[in] points_last end of sequence of (x, y) coordinates of points to be queried
 * @param[out] output_points_first beginning of output range of (x, y) coordinates within the
 * query range
 * @param[in] stream The CUDA stream on which to perform computations and allocate memory.
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
 *
 * [LinkLRAI]: https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
 * "LegacyRandomAccessIterator"
 */
template <class InputIt, class OutputIt, class T>
OutputIt copy_points_in_range(vec_2d<T> vertex_1,
                              vec_2d<T> vertex_2,
                              InputIt points_first,
                              InputIt points_last,
                              OutputIt output_points_first,
                              rmm::cuda_stream_view stream = rmm::cuda_stream_default);

}  // namespace cuspatial

#include <cuspatial/experimental/detail/points_in_range.cuh>
