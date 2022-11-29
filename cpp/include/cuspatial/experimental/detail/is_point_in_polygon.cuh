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

#include <cuspatial/traits.hpp>

#include <cuspatial/detail/utility/floating_point.cuh>

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
  bool is_colinear     = false;
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
      T run          = b.x - a.x;
      T rise         = b.y - a.y;

      // Points on the line segment are the same, so intersection is impossible.
      // This is possible because we allow closed or unclosed polygons.
      T constexpr zero = 0.0;
      if (float_equal(run, zero) && float_equal(rise, zero)) continue;

      T rise_to_point = test_point.y - a.y;

      // colinearity test
      T run_to_point = test_point.x - a.x;
      is_colinear    = float_equal(run * rise_to_point, run_to_point * rise);
      if (is_colinear) { break; }

      y1_flag = a.y > test_point.y;
      if (y1_flag != y0_flag) {
        // Transform the following inequality to avoid division
        //  test_point.x < (run / rise) * rise_to_point + a.x
        auto lhs = (test_point.x - a.x) * rise;
        auto rhs = run * rise_to_point;
        if (lhs < rhs != y1_flag) { point_is_within = not point_is_within; }
      }
      b       = a;
      y0_flag = y1_flag;
    }
    if (is_colinear) {
      point_is_within = false;
      break;
    }
  }

  return point_is_within;
}
}  // namespace detail
}  // namespace cuspatial
