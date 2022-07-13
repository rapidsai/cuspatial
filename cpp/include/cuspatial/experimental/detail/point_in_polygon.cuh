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

#include <cuspatial/detail/utility/traits.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/memory.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {
namespace detail {

template <class Cart2d,
          class Cart2dIdxType,
          class OffsetIteratorA,
          class OffsetIteratorB,
          class Cart2dIt,
          class OffsetItADiffType = typename std::iterator_traits<OffsetIteratorA>::difference_type,
          class OffsetItBDiffType = typename std::iterator_traits<OffsetIteratorB>::difference_type,
          class Cart2dItDiffType  = typename std::iterator_traits<Cart2dIt>::difference_type>
__device__ inline bool is_point_in_polygon(Cart2d const& test_point,
                                           Cart2dIdxType const& poly_idx,
                                           OffsetIteratorA poly_offsets_first,
                                           OffsetItADiffType const& num_polys,
                                           OffsetIteratorB ring_offsets_first,
                                           OffsetItBDiffType const& num_rings,
                                           Cart2dIt poly_points_first,
                                           Cart2dItDiffType const& num_poly_points)
{
  using T = iterator_vec_base_type<Cart2dIt>;

  bool point_is_within = false;
  auto poly_idx_next   = poly_idx + 1;
  auto poly_begin      = poly_offsets_first[poly_idx];
  auto poly_end = (poly_idx_next < num_polys) ? poly_offsets_first[poly_idx_next] : num_rings;

  // for each ring
  for (auto ring_idx = poly_begin; ring_idx < poly_end; ring_idx++) {
    auto ring_idx_next = ring_idx + 1;
    auto ring_begin    = ring_offsets_first[ring_idx];
    auto ring_end =
      (ring_idx_next < num_rings) ? ring_offsets_first[ring_idx_next] : num_poly_points;
    auto ring_len = ring_end - ring_begin;

    // for each line segment, including the segment between the last and first vertex
    for (auto point_idx = 0; point_idx < ring_len; point_idx++) {
      Cart2d const a = poly_points_first[ring_begin + ((point_idx + 0) % ring_len)];
      Cart2d const b = poly_points_first[ring_begin + ((point_idx + 1) % ring_len)];

      bool y_between_ay_by =
        a.y <= test_point.y && test_point.y < b.y;  // is y in range [ay, by) when ay < by?
      bool y_between_by_ay =
        b.y <= test_point.y && test_point.y < a.y;  // is y in range [by, ay) when by < ay?
      bool y_in_bounds = y_between_ay_by || y_between_by_ay;  // is y in range [by, ay]?
      T run            = b.x - a.x;
      T rise           = b.y - a.y;
      T rise_to_point  = test_point.y - a.y;

      if (y_in_bounds && test_point.x < (run / rise) * rise_to_point + a.x) {
        point_is_within = not point_is_within;
      }
    }
  }

  return point_is_within;
}

/**
 * @brief Kernel to test if a point is inside all polygons.
 *
 * The algorithm is based on testing if the point is on one side of the segments in a polygon ring.
 * Each point is tested against all segments in the polygon ring. If the point is on the the "same
 * side" of a segment, it will flip the flag of `point_is_within`. Starting with `point_is_within`
 * as `False`, a point is in the polygon if the flag is flipped odd number of times. Note that for a
 * polygon ring with n vertices, the algorithm tests `n` segments (not `n-1`), including the segment
 * between the last and first vertex.
 */
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
  using Cart2d = iterator_value_type<Cart2dItA>;
  auto idx     = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > num_test_points) { return; }

  int32_t hit_mask = 0;

  Cart2d const test_point = test_points_first[idx];

  // for each polygon
  for (auto poly_idx = 0; poly_idx < num_polys; poly_idx++) {
    bool const point_is_within = is_point_in_polygon(test_point,
                                                     poly_idx,
                                                     poly_offsets_first,
                                                     num_polys,
                                                     ring_offsets_first,
                                                     num_rings,
                                                     poly_points_first,
                                                     num_poly_points);

    hit_mask |= point_is_within << poly_idx;
  }
  result[idx] = hit_mask;
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
  using T = detail::iterator_vec_base_type<Cart2dItA>;

  auto const num_test_points = std::distance(test_points_first, test_points_last);
  auto const num_polys       = std::distance(polygon_offsets_first, polygon_offsets_last);
  auto const num_rings       = std::distance(poly_ring_offsets_first, poly_ring_offsets_last);
  auto const num_poly_points = std::distance(polygon_points_first, polygon_points_last);

  static_assert(detail::is_same_floating_point<T, detail::iterator_vec_base_type<Cart2dItB>>(),
                "Underlying type of Cart2dItA and Cart2dItB must be the same floating point type");
  static_assert(detail::is_same<cartesian_2d<T>,
                                detail::iterator_value_type<Cart2dItA>,
                                detail::iterator_value_type<Cart2dItB>>(),
                "Inputs must be cuspatial::cartesian_2d");

  static_assert(detail::is_integral<detail::iterator_value_type<OffsetIteratorA>,
                                    detail::iterator_value_type<OffsetIteratorB>>(),
                "OffsetIterators must point to integral type.");

  static_assert(std::is_same_v<detail::iterator_value_type<OutputIt>, int32_t>,
                "OutputIt must point to 32 bit integer type.");

  CUSPATIAL_EXPECTS(num_polys <= std::numeric_limits<int32_t>::digits,
                    "Number of polygons cannot exceed 31");

  CUSPATIAL_EXPECTS(num_rings >= num_polys, "Each polygon must have at least one ring");
  CUSPATIAL_EXPECTS(num_poly_points >= num_polys * 4, "Each ring must have at least four vertices");

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
