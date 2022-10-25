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

#include <cuspatial/detail/utility/device_atomics.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/memory.h>

#include <cuda/atomic>

#include <type_traits>

namespace cuspatial {

namespace detail {

template <typename T>
constexpr auto magnitude_squared(T a, T b)
{
  return a * a + b * b;
}

/**
 * @internal
 * @brief computes Hausdorff distance by equally dividing up work on a per-thread basis.
 *
 * Each thread is responsible for computing the distance from a single point in the input against
 * all other points in the input. Because points in the input can originate from different spaces,
 * each thread must know which spaces it is comparing. For the LHS argument, the point is always
 * the same for any given thread and is determined once for that thread using a binary search of
 * the provided space_offsets. Therefore if space 0 contains 10 points, the first 10 threads will
 * know that the LHS space is 0. The 11th thread will know the LHS space is 1, and so on depending
 * on the sizes/offsets of each space. Each thread then loops over each space, and uses an inner
 * loop to loop over each point within that space, thereby knowing the RHS space and RHS point.
 * the thread computes the minimum distance from it's LHS point to _any_ point in the RHS space, as
 * this is the first step to computing Hausdorff distance. The second step of computing Hausdorff
 * distance is to determine the maximum of these minimums, which is done by each thread writing
 * it's minimum to the output using atomicMax. This is done once per thread per RHS space. Once
 * all threads have run to completion, all "maximums of the minumum distances" (aka, directed
 * Hausdorff distances) reside in the output.
 *
 * @tparam T type of coordinate, either float or double.
 * @tparam Index type of indices, e.g. int32_t.
 * @tparam PointsIter Iterator to input points. Points must be of a type that is convertible to
 * `cuspatial::vec_2d<T>`. Must meet the requirements of [LegacyRandomAccessIterator][LinkLRAI] and
 * be device-accessible.
 * @tparam OffsetsIter Iterator to space offsets. Value type must be integral. Must meet the
 * requirements of [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible.
 * @tparam OutputIt Output iterator. Must meet the requirements of
 * [LegacyRandomAccessIterator][LinkLRAI] and be device-accessible and mutable.
 *
 * @param num_points number of total points in the input (sum of points from all spaces)
 * @param points x/y points to compute the distances between
 * @param num_spaces number of spaces in the input
 * @param space_offsets starting position of first point in each space
 * @param results directed Hausdorff distances computed by kernel
 */
template <typename T, typename Index, typename PointIt, typename OffsetIt, typename OutputIt>
__global__ void kernel_hausdorff(
  Index num_points, PointIt points, Index num_spaces, OffsetIt space_offsets, OutputIt results)
{
  using Point = typename std::iterator_traits<PointIt>::value_type;

  // determine the LHS point this thread is responsible for.
  auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  Index const lhs_p_idx = thread_idx;

  if (lhs_p_idx >= num_points) { return; }

  auto const lhs_space_iter =
    thrust::upper_bound(thrust::seq, space_offsets, space_offsets + num_spaces, lhs_p_idx);
  // determine the LHS space this point belongs to.
  Index const lhs_space_idx = thrust::distance(space_offsets, thrust::prev(lhs_space_iter));

  // get the coordinates of this LHS point.
  Point const lhs_p = points[lhs_p_idx];

  // loop over each RHS space, as determined by spa ce_offsets
  for (uint32_t rhs_space_idx = 0; rhs_space_idx < num_spaces; rhs_space_idx++) {
    // determine the begin/end offsets of points contained within this RHS space.
    Index const rhs_p_idx_begin = space_offsets[rhs_space_idx];
    Index const rhs_p_idx_end =
      (rhs_space_idx + 1 == num_spaces) ? num_points : space_offsets[rhs_space_idx + 1];

    // each space must contain at least one point, this initial value is just an identity value to
    // simplify calculations. If a space contains <= 0 points, then this initial value will be
    // written to the output, which can serve as a signal that the input is ill-formed.
    auto min_distance_squared = std::numeric_limits<T>::max();

    // loop over each point in the current RHS space
    for (uint32_t rhs_p_idx = rhs_p_idx_begin; rhs_p_idx < rhs_p_idx_end; rhs_p_idx++) {
      // get the x and y coordinate of this RHS point
      Point const rhs_p = thrust::raw_reference_cast(points[rhs_p_idx]);

      // get distance between the LHS and RHS point
      auto const distance_squared = magnitude_squared(rhs_p.x - lhs_p.x, rhs_p.y - lhs_p.y);

      // remember only smallest distance from this LHS point to any RHS point.
      min_distance_squared = ::min(min_distance_squared, distance_squared);
    }

    // determine the output offset for this pair of spaces (LHS, RHS)
    Index output_idx = lhs_space_idx * num_spaces + rhs_space_idx;

    // use atomicMax to find the maximum of the minimum distance calculated for each space pair.
    atomicMax(&thrust::raw_reference_cast(*(results + output_idx)),
              static_cast<T>(std::sqrt(min_distance_squared)));
  }
}

}  // namespace detail

template <class PointIt, class OffsetIt, class OutputIt>
OutputIt directed_hausdorff_distance(PointIt points_first,
                                     PointIt points_last,
                                     OffsetIt space_offsets_first,
                                     OffsetIt space_offsets_last,
                                     OutputIt distance_first,
                                     rmm::cuda_stream_view stream)
{
  using Point   = typename std::iterator_traits<PointIt>::value_type;
  using Index   = typename std::iterator_traits<OffsetIt>::value_type;
  using T       = typename Point::value_type;
  using OutputT = typename std::iterator_traits<OutputIt>::value_type;

  static_assert(std::is_convertible_v<Point, cuspatial::vec_2d<T>>,
                "Input points must be convertible to cuspatial::vec_2d");
  static_assert(is_floating_point<T, OutputT>(),
                "Hausdorff supports only floating-point coordinates.");
  static_assert(std::is_integral_v<Index>, "Indices must be integral");

  auto const num_points = std::distance(points_first, points_last);
  auto const num_spaces = std::distance(space_offsets_first, space_offsets_last);

  CUSPATIAL_EXPECTS(num_points >= num_spaces, "At least one point is required for each space");
  CUSPATIAL_EXPECTS(num_spaces < (1 << 15), "Total number of spaces must be less than 2^16");

  auto const num_results = num_spaces * num_spaces;

  if (num_results > 0) {
    // Due to hausdorff kernel using `atomicMax` for output, the output must be initialized to <= 0
    // here the output is being initialized to -1, which should always be overwritten. If -1 is
    // found in the output, there is a bug where the output is not being written to in the hausdorff
    // kernel.
    thrust::fill_n(rmm::exec_policy(stream), distance_first, num_results, -1);

    auto const threads_per_block = 64;
    auto const num_tiles         = (num_points + threads_per_block - 1) / threads_per_block;

    detail::kernel_hausdorff<T, decltype(num_points)>
      <<<num_tiles, threads_per_block, 0, stream.value()>>>(
        num_points, points_first, num_spaces, space_offsets_first, distance_first);

    CUSPATIAL_CUDA_TRY(cudaGetLastError());
  }

  return distance_first + num_results;
}

}  // namespace cuspatial
