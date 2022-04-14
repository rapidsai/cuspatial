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
#include <cuspatial/types.hpp>

#include <cudf/detail/utilities/device_atomics.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <limits>
#include <memory>
#include <type_traits>

namespace cuspatital {
namespace detail {

template<typename T, typename ...Ts>
constexpr bool is_same() {
    return std::conjunction_v<std::is_same<T, Ts>...>;
}

template<typename ...Ts>
constexpr bool is_floating_point() {
    return std::conjunction_v<std::is_floating_point<Ts>...>;
}

template <typename T>
double __device__ point_to_segment_distance(cuspatial::cart_2d<T> const& C,
                                            cuspatial::cart_2d<T> const& A,
                                            cuspatial::cart_2d<T> const& B)
{
  // Subject 1.02 of https://www.inf.pucrs.br/~pinho/CG/faq.html
  // Project the point to the segment, if it lands on the segment,
  // the distance is the length of proejction, otherwise it's the
  // length to one of the end points.

  double L_squared = (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
  if (L_squared == 0) { return hypot(C.x - A.x, C.y - A.y); }
  double r = ((C.x - A.x) * (B.x - A.x) + (C.y - A.y) * (B.y - A.y)) / L_squared;
  if (r <= 0 or r >= 1) {
    return std::min(hypot(C.x - A.x, C.y - A.y), hypot(C.x - B.x, C.y - B.y));
  }
  double Px = A.x + r * (B.x - A.x);
  double Py = A.y + r * (B.y - A.y);
  return hypot(C.x - Px, C.y - Py);
}

template <typename T>
double __device__ segment_distance_no_intersect(cuspatial::cart_2d<T> const& A,
                                                cuspatial::cart_2d<T> const& B,
                                                cuspatial::cart_2d<T> const& C,
                                                cuspatial::cart_2d<T> const& D)
{
  return std::min(std::min(point_to_segment_distance(A, C, D), point_to_segment_distance(B, C, D)),
                  std::min(point_to_segment_distance(C, A, B), point_to_segment_distance(D, A, B)));
}

/**
 * @brief Computes shortest distance between two segments.
 *
 * If two segment intersects, distance is 0.
 */
template <typename T>
double __device__ segment_distance(cuspatial::cart_2d<T> const& A,
                                   cuspatial::cart_2d<T> const& B,
                                   cuspatial::cart_2d<T> const& C,
                                   cuspatial::cart_2d<T> const& D)
{
  // Subject 1.03 of https://www.inf.pucrs.br/~pinho/CG/faq.html
  // Construct a parametrized ray of AB and CD, solve for the parameters.
  // If both parameters are within [0, 1], the intersection exists.

  double r_denom = (B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x);
  double r_numer = (A.y - C.y) * (D.x - C.x) - (A.x - C.x) * (D.y - C.y);
  if (r_denom == 0) {
    if (r_numer == 0) { return 0.0; }  // Segments coincides
    // Segments parallel
    return segment_distance_no_intersect(A, B, C, D);
  }
  double r = r_numer / r_denom;
  double s = ((A.y - C.y) * (B.x - A.x) - (A.x - C.x) * (B.y - A.x)) /
             ((B.x - A.x) * (D.y - C.y) - (B.y - A.y) * (D.x - C.x));
  if (r >= 0 and r <= 1 and s >= 0 and s <= 1) { return 0.0; }
  return segment_distance_no_intersect(A, B, C, D);
}

template <typename Cart2dItA,
          typename Cart2dItB,
          typename OffsetIterator,
          typename OutputIterator,
          typename T = typename std::iterator_traits<Cart2dItA>::value_type>
void __global__ kernel(OffsetIterator linestring1_offsets_begin,
                       OffsetIterator linestring1_offsets_end,
                       Cart2dItA linestring1_points_begin,
                       Cart2dItA linestring1_points_end,
                       OffsetIterator linestring2_offsets_begin,
                       Cart2dItB linestring2_points_begin,
                       Cart2dItB linestring2_points_end,
                       OutputIterator min_distances)
{
  auto const p1Idx = threadIdx.x + blockIdx.x * blockDim.x;
  cudf::size_type const num_linestrings =
    thrust::distance(linestring1_offsets_begin, linestring1_offsets_end);
  cudf::size_type const linestring1_num_points =
    thrust::distance(linestring1_points_begin, linestring1_points_end);
  cudf::size_type const linestring2_num_points =
    thrust::distance(linestring2_points_begin, linestring2_points_end);

  if (p1Idx >= linestring1_num_points) { return; }

  cudf::size_type const linestring_idx =
    thrust::distance(
      linestring1_offsets_begin,
      thrust::upper_bound(thrust::seq, linestring1_offsets_begin, linestring1_offsets_end, p1Idx)) -
    1;

  cudf::size_type ls1End =
    (linestring_idx == (num_linestrings - 1) ? (linestring1_num_points)
                                             : *(linestring1_offsets_begin + linestring_idx + 1)) -
    1;

  if (p1Idx == ls1End) {
    // Current point is the end point of the line string.
    return;
  }

  cudf::size_type ls2Start = *(linestring2_offsets_begin + linestring_idx);
  cudf::size_type ls2End =
    (linestring_idx == (num_linestrings - 1) ? linestring2_num_points
                                             : *(linestring2_offsets_begin + linestring_idx + 1)) -
    1;

  cuspatial::cart_2d<T> const& A = linestring1_points_begin[p1Idx];
  cuspatial::cart_2d<T> const& B = linestring1_points_begin[p1Idx + 1];

  double min_distance = std::numeric_limits<double>::max();
  for (cudf::size_type p2Idx = ls2Start; p2Idx < ls2End; p2Idx++) {
    cuspatial::cart_2d<T> const& C = linestring2_points_begin[p2Idx];
    cuspatial::cart_2d<T> const& D = linestring2_points_begin[p2Idx + 1];
    min_distance = std::min(min_distance, segment_distance(A, B, C, D));
  }
  atomicMin(min_distances + linestring_idx, static_cast<T>(min_distance));
}


} // namespace detail

/**
 * @brief Check if every linestring in the input contains at least 2 end points.
 */
template<typename OffsetIterator, typename Cart2dIt>
bool validate_linestring(OffsetIterator linestring_offsets_first,
                         cudf::size_type num_linestring_pairs,
                         Cart2dIt linestring_points_x_first,
                         cudf::size_type num_points,
                         rmm::cuda_stream_view stream)
{
  if (num_linestring_pairs == 1) {
    return num_points >= 2;
  }

  return thrust::reduce(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(cudf::size_type{0}),
    thrust::make_counting_iterator(num_linestring_pairs),
    true,
    [linestring_offsets_first,
     num_linestring_pairs,
     num_points] __device__(bool prev, cudf::size_type i) {
      cudf::size_type begin = linestring_offsets_first[i];
      cudf::size_type end   = i == num_linestring_pairs ? num_points : linestring_offsets_first[i + 1];
      return prev && (end - begin);
    });
}

template<class Cart2dItA,
         class Cart2dItB,
         class OffsetIterator,
         class OutputIt,
         class Cart2d,
         class T>
 void pairwise_linestring_distance(
  OffsetIterator linestring1_offsets_first,
  OffsetIterator linestring1_offsets_last,
  Cart2dItA linestring1_points_first,
  Cart2dItA linestring1_points_last,
  OffsetIterator linestring2_offsets_first,
  Cart2dItB linestring2_points_first,
  Cart2dItB linestring2_points_last,
  OutputIt distances_first,
  rmm::cuda_stream_view stream) {
        using Cart2dB = typename std::iterator_traits<Cart2dItB>::value_type;
        static_assert(
            detail::is_same<cuspatial::cart_2d, Cart2d, Cart2dB>(), "Inputs must be cuspatial::cart_2d"
        );
        static_assert(
            detail::is_floating_point<T, typename Cart2dB::value_type, typename std::iterator_traits<OutputIt>::value_type>(),
            "Inputs must be floating point types."
        );

        auto const num_string_pairs = thrust::distance(linestring1_offsets_first, linestring1_offsets_last);
        auto const num_linestring1_points = thrust::distance(linestring1_points_first, linestring1_points_last);
        auto const num_linestring2_points = thrust::distance(linestring2_points_first, linestring2_points_last);

        CUSPATIAL_EXPECTS(validate_linestring(
            linestring1_offsets_first, num_string_pairs, linestring1_points_first, num_linestring1_points, stream),
                            "Each item of linestring1 should contain at least 2 end points.");
        CUSPATIAL_EXPECTS(validate_linestring(
            linestring2_offsets_first, num_string_pairs, linestring2_points_first, num_linestring2_points, stream),
                            "Each item of linestring2 should contain at least 2 end points.");


        thrust::fill(rmm::exec_policy(stream),
                    distances_first,
                    distances_first + num_string_pairs,
                    std::numeric_limits<T>::max());
        
        std::size_t const threads_per_block = 64;
        std::size_t const num_blocks = (num_linestring1_points + threads_per_block - 1) / threads_per_block;

        kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
            linestring1_offsets_first,
            linestring1_offsets_last,
            linestring1_points_first,
            linestring1_points_last,
            linestring2_offsets_first,
            linestring2_points_first,
            linestring2_points_last,
            distances_first
        );

        CUSPATIAL_CUDA_TRY(cudaGetLastError());
  }

} // namespace cuspatial
