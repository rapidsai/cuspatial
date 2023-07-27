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

#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/geometry_collection/multipoint_ref.cuh>
#include <cuspatial/traits.hpp>

#include <cuspatial/detail/utility/floating_point.cuh>
#include <cuspatial/geometry/polygon_ref.cuh>

#include <thrust/swap.h>

namespace cuspatial {
namespace detail {

/**
 * @brief Test if a point is inside a polygon.
 *
 * Implemented based on Eric Haines's crossings-multiply algorithm:
 * See "Crossings test" section of http://erich.realtimerendering.com/ptinpoly/
 * The improvement in addenda is also adopted to remove divisions in this kernel.
 *
 * @tparam T type of coordinate
 * @tparam PolygonRef polygon_ref type
 * @param test_point point to test for point in polygon
 * @param polygon polygon to test for point in polygon
 * @return boolean to indicate if point is inside the polygon.
 * `false` if point is on the edge of the polygon.
 */
template <typename T, class PolygonRef>
__device__ inline bool is_point_in_polygon(vec_2d<T> const& test_point, PolygonRef const& polygon)
{
  bool point_is_within = false;
  bool point_on_edge   = false;
  for (auto ring : polygon) {
    auto last_segment = ring.segment(ring.num_segments() - 1);

    auto b       = last_segment.v2;
    bool y0_flag = b.y > test_point.y;
    bool y1_flag;
    auto ring_points = multipoint_ref{ring.point_begin(), ring.point_end()};
    for (vec_2d<T> a : ring_points) {
      // for each line segment, including the segment between the last and first vertex
      T run  = b.x - a.x;
      T rise = b.y - a.y;

      // Points on the line segment are the same, so intersection is impossible.
      // This is possible because we allow closed or unclosed polygons.
      T constexpr zero = 0.0;
      if (float_equal(run, zero) && float_equal(rise, zero)) continue;

      T rise_to_point = test_point.y - a.y;
      T run_to_point  = test_point.x - a.x;

      // point-on-edge test
      bool is_collinear = float_equal(run * rise_to_point, run_to_point * rise);
      if (is_collinear) {
        T minx = a.x;
        T maxx = b.x;
        if (minx > maxx) thrust::swap(minx, maxx);
        if (minx <= test_point.x && test_point.x <= maxx) {
          point_on_edge = true;
          break;
        }
      }

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
    if (point_on_edge) {
      point_is_within = false;
      break;
    }
  }

  return point_is_within;
}

}  // namespace detail
}  // namespace cuspatial
