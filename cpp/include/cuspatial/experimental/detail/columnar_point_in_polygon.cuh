/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuspatial/detail/utility/validation.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/detail/is_point_in_polygon.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/memory.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {
namespace detail {

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OutputIt,
          class Cart2dItADiffType = typename std::iterator_traits<Cart2dItA>::difference_type,
          class Cart2dItBDiffType = typename std::iterator_traits<Cart2dItB>::difference_type,
          class OffsetItADiffType = typename std::iterator_traits<OffsetIteratorA>::difference_type,
          class OffsetItBDiffType = typename std::iterator_traits<OffsetIteratorB>::difference_type>
__global__ void columnar_point_in_polygon_kernel(Cart2dItA test_points_first,
                                                 Cart2dItADiffType const num_test_points,
                                                 OffsetIteratorA poly_offsets_first,
                                                 OffsetItADiffType const num_polys,
                                                 OffsetIteratorB ring_offsets_first,
                                                 OffsetItBDiffType const num_rings,
                                                 Cart2dItB poly_points_first,
                                                 Cart2dItBDiffType const num_poly_points,
                                                 OutputIt result)
{
  using Cart2d     = iterator_value_type<Cart2dItA>;
  using OffsetType = iterator_value_type<OffsetIteratorA>;

  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_test_points) { return; }

  Cart2d const test_point = test_points_first[idx];

  // for each polygon
  for (auto poly_idx = 0; poly_idx < num_polys; poly_idx++) {
    auto poly_idx_next    = poly_idx + 1;
    OffsetType poly_begin = poly_offsets_first[poly_idx];
    OffsetType poly_end =
      (poly_idx_next < num_polys) ? poly_offsets_first[poly_idx_next] : num_rings;

    bool const point_is_within = is_point_in_polygon(test_point,
                                                     poly_begin,
                                                     poly_end,
                                                     ring_offsets_first,
                                                     num_rings,
                                                     poly_points_first,
                                                     num_poly_points);

    result[num_test_points * poly_idx + idx] = point_is_within;
  }
}

}  // namespace detail

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OutputIt>
OutputIt columnar_point_in_polygon(Cart2dItA test_points_first,
                                   Cart2dItA test_points_last,
                                   OffsetIteratorA polygon_offsets_first,
                                   OffsetIteratorA polygon_offsets_last,
                                   OffsetIteratorB poly_ring_offsets_first,
                                   OffsetIteratorB poly_ring_offsets_last,
                                   Cart2dItB polygon_points_first,
                                   Cart2dItB polygon_points_last,
                                   OutputIt output,
                                   rmm::cuda_stream_view stream)
{
  using T = iterator_vec_base_type<Cart2dItA>;

  static_assert(is_same_floating_point<T, iterator_vec_base_type<Cart2dItB>>(),
                "Underlying type of Cart2dItA and Cart2dItB must be the same floating point type");
  static_assert(
    is_same<vec_2d<T>, iterator_value_type<Cart2dItA>, iterator_value_type<Cart2dItB>>(),
    "Inputs must be cuspatial::vec_2d");

  static_assert(cuspatial::is_integral<iterator_value_type<OffsetIteratorA>,
                                       iterator_value_type<OffsetIteratorB>>(),
                "OffsetIterators must point to integral type.");

  auto const num_test_points = std::distance(test_points_first, test_points_last);
  auto const num_polys       = std::distance(polygon_offsets_first, polygon_offsets_last) - 1;
  auto const num_rings       = std::distance(poly_ring_offsets_first, poly_ring_offsets_last) - 1;
  auto const num_poly_points = std::distance(polygon_points_first, polygon_points_last);

  if (num_test_points > 0) {
    CUSPATIAL_EXPECTS_VALID_POLYGON_SIZES(
      num_poly_points,
      std::distance(polygon_offsets_first, polygon_offsets_last),
      std::distance(poly_ring_offsets_first, poly_ring_offsets_last));

    auto [threads_per_block, num_blocks] = grid_1d(num_test_points);

    detail::columnar_point_in_polygon_kernel<<<num_blocks, threads_per_block, 0, stream.value()>>>(
      test_points_first,
      num_test_points,
      polygon_offsets_first,
      num_polys,
      poly_ring_offsets_first,
      num_rings,
      polygon_points_first,
      num_poly_points,
      output);
    CUSPATIAL_CHECK_CUDA(stream.value());
  }

  return output + num_test_points * num_polys;
}

}  // namespace cuspatial
