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

#include <cuspatial/detail/utility/floating_point.cuh>
#include <cuspatial/experimental/geometry/segment.cuh>
#include <cuspatial/vec_2d.hpp>

#include <thrust/optional.h>
#include <thrust/pair.h>
#include <thrust/swap.h>
#include <thrust/tuple.h>

namespace cuspatial {
namespace detail {

/**
 * @internal
 * @brief Get the index that is one-past the end point of linestring at @p linestring_idx
 *
 * @note The last endpoint of the linestring is not included in the offset array, thus
 * @p num_points is returned.
 */
template <typename SizeType, typename OffsetIterator>
__forceinline__ SizeType __device__
endpoint_index_of_linestring(SizeType const& linestring_idx,
                             OffsetIterator const& linestring_offsets_begin,
                             SizeType const& num_linestrings,
                             SizeType const& num_points)
{
  auto const is_end = linestring_idx == (num_linestrings - 1);
  return (is_end ? num_points : *(linestring_offsets_begin + linestring_idx + 1)) - 1;
}

/**
 * @internal
 * @brief Computes shortest distance and nearest point between @p c and segment
 * ab
 */
template <typename T>
__forceinline__ thrust::tuple<T, vec_2d<T>> __device__
point_to_segment_distance_squared_nearest_point(vec_2d<T> const& c,
                                                vec_2d<T> const& a,
                                                vec_2d<T> const& b)
{
  auto ab        = b - a;
  auto ac        = c - a;
  auto l_squared = dot(ab, ab);
  if (float_equal(l_squared, T{0})) { return thrust::make_tuple(dot(ac, ac), a); }
  auto r  = dot(ac, ab);
  auto bc = c - b;
  // If the projection of `c` is outside of segment `ab`, compute point-point distance.
  if (r <= 0 or r >= l_squared) {
    auto dac = dot(ac, ac);
    auto dbc = dot(bc, bc);
    return dac < dbc ? thrust::make_tuple(dac, a) : thrust::make_tuple(dbc, b);
  }
  auto p  = a + (r / l_squared) * ab;
  auto pc = c - p;
  return thrust::make_tuple(dot(pc, pc), p);
}

/**
 * @internal
 * @brief Computes shortest distance between @p c and segment ab
 */
template <typename T>
__forceinline__ T __device__ point_to_segment_distance_squared(vec_2d<T> const& c,
                                                               vec_2d<T> const& a,
                                                               vec_2d<T> const& b)
{
  [[maybe_unused]] auto [distance_squared, _] =
    point_to_segment_distance_squared_nearest_point(c, a, b);
  return distance_squared;
}

/**
 * @internal
 * @brief Computes shortest distance between two segments (ab and cd) that don't intersect.
 */
template <typename T>
__forceinline__ T __device__ segment_distance_no_intersect_or_colinear(vec_2d<T> const& a,
                                                                       vec_2d<T> const& b,
                                                                       vec_2d<T> const& c,
                                                                       vec_2d<T> const& d)
{
  auto dist_sqr = min(
    min(point_to_segment_distance_squared(a, c, d), point_to_segment_distance_squared(b, c, d)),
    min(point_to_segment_distance_squared(c, a, b), point_to_segment_distance_squared(d, a, b)));
  return dist_sqr;
}

/**
 * @internal
 * @brief Computes shortest distance between two segments.
 *
 * If two segments intersect, the distance is 0. Otherwise compute the shortest point
 * to segment distance.
 */
template <typename T>
__forceinline__ T __device__ squared_segment_distance(vec_2d<T> const& a,
                                                      vec_2d<T> const& b,
                                                      vec_2d<T> const& c,
                                                      vec_2d<T> const& d)
{
  auto ab    = b - a;
  auto cd    = d - c;
  auto denom = det(ab, cd);

  if (float_equal(denom, T{0})) {
    // Segments parallel or collinear
    return segment_distance_no_intersect_or_colinear(a, b, c, d);
  }

  auto ac               = c - a;
  auto r_numer          = det(ac, cd);
  auto denom_reciprocal = 1 / denom;
  auto r                = r_numer * denom_reciprocal;
  auto s                = det(ac, ab) * denom_reciprocal;
  if (r >= 0 and r <= 1 and s >= 0 and s <= 1) { return 0.0; }
  return segment_distance_no_intersect_or_colinear(a, b, c, d);
}

/**
 * @internal
 * @brief Given two collinear or parallel segments, return their potential overlapping segment
 *
 * @p a, @p b, @p c, @p d refer to end points of segment ab and cd.
 * @p center is the geometric center of the segments, used to decondition the coordinates.
 *
 * @return optional end points of overlapping segment
 */
template <typename T>
__forceinline__ thrust::optional<segment<T>> __device__ collinear_or_parallel_overlapping_segments(
  vec_2d<T> a, vec_2d<T> b, vec_2d<T> c, vec_2d<T> d, vec_2d<T> center = vec_2d<T>{})
{
  auto ab = b - a;
  auto ac = c - a;

  // Parallel
  if (not float_equal(det(ab, ac), T{0})) return thrust::nullopt;

  // Must be on the same line, sort the endpoints
  if (b < a) thrust::swap(a, b);
  if (d < c) thrust::swap(c, d);

  // Test if not overlap
  if (b < c || d < a) return thrust::nullopt;

  // Compute smallest interval between the segments
  auto e0 = a > c ? a : c;
  auto e1 = b < d ? b : d;

  // Decondition the coordinates
  return segment<T>{e0 + center, e1 + center};
}

/**
 * @internal
 * @brief Primitive to compute intersections between two segments
 * Two segments can intersect at a point, overlap at a segment, or be disjoint.
 *
 * @return A pair of optional intersecting point and optional overlapping segment
 */
template <typename T>
__forceinline__ thrust::pair<thrust::optional<vec_2d<T>>, thrust::optional<segment<T>>> __device__
segment_intersection(segment<T> const& segment1, segment<T> const& segment2)
{
  // Condition the coordinates to avoid large floating point error
  auto center = midpoint(segment1.center(), segment2.center());
  auto [a, b] = segment1.translate(-center);
  auto [c, d] = segment2.translate(-center);

  auto ab = b - a;
  auto cd = d - c;

  auto denom = det(ab, cd);

  if (float_equal(denom, T{0})) {
    // Segments parallel or collinear
    return {thrust::nullopt, collinear_or_parallel_overlapping_segments(a, b, c, d, center)};
  }

  auto ac               = c - a;
  auto r_numer          = det(ac, cd);
  auto denom_reciprocal = 1 / denom;
  auto r                = r_numer * denom_reciprocal;
  auto s                = det(ac, ab) * denom_reciprocal;
  if (r >= 0 and r <= 1 and s >= 0 and s <= 1) {
    auto p = a + r * ab;
    // Decondition the coordinates
    return {p + center, thrust::nullopt};
  }
  return {thrust::nullopt, thrust::nullopt};
}

/**
 * @brief Given a point and a segment, returns true if point is on the segment.
 */
template <typename T>
bool __device__ is_point_on_segment(segment<T> const& segment, vec_2d<T> const& c)
{
  auto [a, b] = segment;
  auto ab     = b - a;
  auto ac     = c - a;

  if (not float_equal(det(ab, ac), T{0})) return false;

  if (b < a) thrust::swap(a, b);
  return a <= c && c <= b;
}

/*
 * @brief Given two segments, if they are mergable, return the merged result. Otherwise return
 * nullopt.
 */
template <typename T>
thrust::optional<segment<T>> __device__ maybe_merge_segments(segment<T> const& segment1,
                                                             segment<T> const& segment2)
{
  // Condition the coordinates to avoid large floating point error
  auto center = midpoint(segment1.center(), segment2.center());
  auto [a, b] = segment1.translate(-center);
  auto [c, d] = segment2.translate(-center);

  auto ab = b - a;
  auto cd = d - c;

  if (not float_equal(det(ab, cd), T{0})) return thrust::nullopt;
  auto ac = c - a;
  if (not float_equal(det(ab, ac), T{0})) return thrust::nullopt;

  // Must be on the same line, sort the endpoints
  if (b < a) thrust::swap(a, b);
  if (d < c) thrust::swap(c, d);

  // Test if not overlap
  if (b < c || d < a) return thrust::nullopt;

  // Compute largest interval between the segments
  auto e0 = a < c ? a : c;
  auto e1 = b > d ? b : d;

  // Decondition the coordinates
  return segment<T>{e0, e1}.translate(center);
}

}  // namespace detail
}  // namespace cuspatial
