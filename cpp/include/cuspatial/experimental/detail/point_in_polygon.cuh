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

#include <thrust/memory.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {
namespace detail {

/**
 * @brief Kernel to test if a point is inside a polygon.
 *
 * Implemented based on Eric Haines's crossings-multiply algorithm:
 * See "Crossings test" section of http://erich.realtimerendering.com/ptinpoly/
 * The improvement in addenda is also addopted to remove divisions in this kernel.
 *
 * TODO: the ultimate goal of refactoring this as independent function is to remove
 * src/utility/point_in_polygon.cuh and its usage in quadtree_point_in_polygon.cu. It isn't
 * possible today without further work to refactor quadtree_point_in_polygon into header only
 * API.
 */
template <class Cart2d,
          class OffsetType,
          class OffsetIterator,
          class Cart2dIt,
          class OffsetItDiffType = typename std::iterator_traits<OffsetIterator>::difference_type,
          class Cart2dItDiffType = typename std::iterator_traits<Cart2dIt>::difference_type>
__device__ inline bool is_point_in_polygon(Cart2d const& test_point,
                                           OffsetType poly_begin,
                                           OffsetType poly_end,
                                           OffsetIterator ring_offsets_first,
                                           OffsetItDiffType const& num_rings,
                                           Cart2dIt poly_points_first,
                                           Cart2dItDiffType const& num_poly_points)
{
  using T = iterator_vec_base_type<Cart2dIt>;

  bool point_is_within = false;
  // for each ring
  for (auto ring_idx = poly_begin; ring_idx < poly_end; ring_idx++) {
    int32_t ring_idx_next = ring_idx + 1;
    int32_t ring_begin    = ring_offsets_first[ring_idx];
    int32_t ring_end =
      (ring_idx_next < num_rings) ? ring_offsets_first[ring_idx_next] : num_poly_points;

    Cart2d b     = poly_points_first[ring_end - 1];
    bool y0_flag = b.y > test_point.y;
    bool y1_flag;
    // for each line segment, including the segment between the last and first vertex
    for (auto point_idx = ring_begin; point_idx < ring_end; point_idx++) {
      Cart2d const a = poly_points_first[point_idx];
      y1_flag        = a.y > test_point.y;
      if (y1_flag != y0_flag) {
        T run           = b.x - a.x;
        T rise          = b.y - a.y;
        T rise_to_point = test_point.y - a.y;

        // Transform the following inequality to avoid division
        //  test_point.x < (run / rise) * rise_to_point + a.x
        auto lhs = (test_point.x - a.x) * rise;
        auto rhs = run * rise_to_point;
        if ((rise > 0 && lhs < rhs) || (rise < 0 && lhs > rhs))
          point_is_within = not point_is_within;
      }
      b       = a;
      y0_flag = y1_flag;
    }
  }

  return point_is_within;
}

template <class Cart2dItA,
          class Cart2dItB,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class OutputIt,
          class Cart2dItADiffType = typename std::iterator_traits<Cart2dItA>::difference_type,
          class Cart2dItBDiffType = typename std::iterator_traits<Cart2dItB>::difference_type,
          class OffsetItADiffType = typename std::iterator_traits<OffsetIteratorA>::difference_type,
          class OffsetItBDiffType = typename std::iterator_traits<OffsetIteratorB>::difference_type>
__global__ void point_in_polygon_kernel(Cart2dItA test_points_first,
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
OutputIt point_in_polygon(Cart2dItA test_points_first,
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

  auto const num_test_points = std::distance(test_points_first, test_points_last);
  auto const num_polys       = std::distance(polygon_offsets_first, polygon_offsets_last);
  auto const num_rings       = std::distance(poly_ring_offsets_first, poly_ring_offsets_last);
  auto const num_poly_points = std::distance(polygon_points_first, polygon_points_last);

  static_assert(is_same_floating_point<T, iterator_vec_base_type<Cart2dItB>>(),
                "Underlying type of Cart2dItA and Cart2dItB must be the same floating point type");
  static_assert(
    is_same<vec_2d<T>, iterator_value_type<Cart2dItA>, iterator_value_type<Cart2dItB>>(),
    "Inputs must be cuspatial::vec_2d");

  static_assert(cuspatial::is_integral<iterator_value_type<OffsetIteratorA>,
                                       iterator_value_type<OffsetIteratorB>>(),
                "OffsetIterators must point to integral type.");

  // TODO Assert it points to a column, is that an OffsetIterator?
  // static_assert(cuspatial::is_integral<iterator_value_type<OutputIt>>(),
  //             "OutputIt must point to integral type.");

  CUSPATIAL_EXPECTS(num_polys <= std::numeric_limits<int32_t>::digits,
                    "Number of polygons cannot exceed 31");

  CUSPATIAL_EXPECTS(num_rings >= num_polys, "Each polygon must have at least one ring");
  CUSPATIAL_EXPECTS(num_poly_points >= num_polys * 4, "Each ring must have at least four vertices");

  // TODO: introduce a validation function that checks the rings of the polygon are
  // actually closed. (i.e. the first and last vertices are the same)

  auto constexpr block_size = 256;
  auto const num_blocks     = (num_test_points + block_size - 1) / block_size;

  detail::point_in_polygon_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    test_points_first,
    num_test_points,
    polygon_offsets_first,
    num_polys,
    poly_ring_offsets_first,
    num_rings,
    polygon_points_first,
    num_poly_points,
    output);
  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return output + num_test_points;
}

}  // namespace cuspatial
