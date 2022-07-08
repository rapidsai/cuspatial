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

template <class Cart2dItA, class Cart2dItB, class OffsetIteratorA, class OffsetIteratorB, class OutputIt>
__global__ void point_in_polygon_kernel(Cart2dItA test_points_begin,
                                        int32_t const num_test_points,
                                        OffsetIteratorA poly_offsets_begin,
                                        int32_t const num_polys,
                                        OffsetIteratorB ring_offsets_begin,
                                        int32_t const num_rings,
                                        Cart2dItB poly_points_begin,
                                        int32_t const num_poly_points,
                                        OutputIt result)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  auto idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx > num_test_points) { return; }

  int32_t hit_mask = 0;

  auto const test_point = thrust::raw_reference_cast(test_points_begin[idx]);

  // for each polygon
  for (auto poly_idx = 0; poly_idx < num_polys; poly_idx++) {
    auto poly_idx_next = poly_idx + 1;
    auto poly_begin    = poly_offsets_begin[poly_idx];
    auto poly_end = (poly_idx_next < num_polys) ? poly_offsets_begin[poly_idx_next] : num_rings;

    bool point_is_within = false;

    // for each ring
    for (auto ring_idx = poly_begin; ring_idx < poly_end; ring_idx++) {
      auto ring_idx_next = ring_idx + 1;
      auto ring_begin    = ring_offsets_begin[ring_idx];
      auto ring_end =
        (ring_idx_next < num_rings) ? ring_offsets_begin[ring_idx_next] : num_poly_points;
      auto ring_len = ring_end - ring_begin;

      // for each line segment
      for (auto point_idx = 0; point_idx < ring_len; point_idx++) {
        auto const a =
          thrust::raw_reference_cast(poly_points_begin[ring_begin + ((point_idx + 0) % ring_len)]);
        auto const b =
          thrust::raw_reference_cast(poly_points_begin[ring_begin + ((point_idx + 1) % ring_len)]);

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

    hit_mask |= point_is_within << poly_idx;
  }
  result[idx] = hit_mask;
}

}  // namespace detail

template <class Cart2dItA, class Cart2dItB, class OffsetIteratorA, class OffsetIteratorB , class OutputIt>
OutputIt point_in_polygon(Cart2dItA points_begin,
                          Cart2dItA points_end,
                          OffsetIteratorA polygon_offsets_begin,
                          OffsetIteratorA polygon_offsets_end,
                          OffsetIteratorB ring_offsets_begin,
                          OffsetIteratorB ring_offsets_end,
                          Cart2dItB polygon_points_begin,
                          Cart2dItB polygon_points_end,
                          OutputIt output,
                          rmm::cuda_stream_view stream)
{
  using T = typename std::iterator_traits<Cart2dItA>::value_type::value_type;

  auto const num_test_points = std::distance(points_begin, points_end);
  auto const num_polys       = std::distance(polygon_offsets_begin, polygon_offsets_end);
  auto const num_rings       = std::distance(ring_offsets_begin, ring_offsets_end);
  auto const num_poly_points = std::distance(polygon_points_begin, polygon_points_end);

  static_assert(
    detail::
      is_same_floating_point<T, typename std::iterator_traits<Cart2dItB>::value_type::value_type>(),
    "Underlying type of Cart2dItA and Cart2dItB must be the same floating point type");
  static_assert(detail::is_same<cartesian_2d<T>,
                                typename std::iterator_traits<Cart2dItA>::value_type,
                                typename std::iterator_traits<Cart2dItB>::value_type>(),
                "Inputs must be cuspatial::cartesian_2d");

  static_assert(std::is_integral_v<typename std::iterator_traits<OffsetIteratorA>::value_type> && std::is_integral_v<typename std::iterator_traits<OffsetIteratorB>::value_type>,
                "OffsetIterator must point to integral type.");

  static_assert(std::is_same_v<typename std::iterator_traits<OutputIt>::value_type, int32_t>,
                "OutputIt must point to 32 bit integer type.");

  CUSPATIAL_EXPECTS(num_polys <= std::numeric_limits<int32_t>::digits,
                    "Number of polygons cannot exceed 31");

  CUSPATIAL_EXPECTS(num_rings >= num_polys, "Each polygon must have at least one ring");
  CUSPATIAL_EXPECTS(num_poly_points >= num_polys * 4, "Each ring must have at least four vertices");

  auto constexpr block_size = 256;
  auto const num_blocks     = (num_test_points + block_size - 1) / block_size;

  detail::point_in_polygon_kernel<<<num_blocks, block_size, 0, stream.value()>>>(
    points_begin,
    num_test_points,
    polygon_offsets_begin,
    num_polys,
    ring_offsets_begin,
    num_rings,
    polygon_points_begin,
    num_poly_points,
    output);
  CUSPATIAL_CUDA_TRY(cudaGetLastError());

  return output + num_test_points;
}

}  // namespace cuspatial
