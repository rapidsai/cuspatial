#pragma once

#include <cudf/types.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>

#include <thrust/distance.h>
#include <thrust/binary_search.h>

#include <limits>

namespace cuspatial {

enum class DISTANCE_KIND {
  SHORTEST,
  HAUSDORFF
};

namespace detail {


  using size_type = cudf::size_type;

  template <typename T>
  constexpr auto magnitude_squared(T a, T b)
  {
    return a * a + b * b;
  }
  
  /**
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
   * @tparam T type of coordinate, either float or double.
   *
   * @param num_points number of total points in the input (sum of points from all spaces)
   * @param xs x coordinates
   * @param ys y coordinates
   * @param num_spaces number of spaces in the input
   * @param space_offsets starting position of each first point in each space
   * @param results directed Hausdorff distances computed by kernel
   * @return
   */
   template <typename T, DISTANCE_KIND kind>
   __global__ void distances_kernel(size_type num_points,
                                    T const* xs,
                                    T const* ys,
                                    size_type num_spaces,
                                    size_type const* space_offsets,
                                    T* results)
   {
     // determine the LHS point this thread is responsible for.
     auto const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
     auto const lhs_p_idx  = thread_idx;
   
     if (lhs_p_idx >= num_points) { return; }
   
     // determine the LHS space this point belongs to.
     auto const lhs_space_idx =
       thrust::distance(
         space_offsets,
         thrust::upper_bound(thrust::seq, space_offsets, space_offsets + num_spaces, lhs_p_idx)) -
       1;
   
     // get the x and y coordinate of this LHS point.
     auto const lhs_p_x = xs[lhs_p_idx];
     auto const lhs_p_y = ys[lhs_p_idx];
   
     // loop over each RHS space, as determined by space_offsets
     for (uint32_t rhs_space_idx = 0; rhs_space_idx < num_spaces; rhs_space_idx++) {
       // determine the begin/end offsets of points contained within this RHS space.
       auto const rhs_p_idx_begin = space_offsets[rhs_space_idx];
       auto const rhs_p_idx_end =
         (rhs_space_idx + 1 == num_spaces) ? num_points : space_offsets[rhs_space_idx + 1];
   
       // each space must contain at least one point, this initial value is just an identity value to
       // simplify calculations. If a space contains <= 0 points, then this initial value will be
       // written to the output, which can serve as a signal that their input is ill-formed.
       auto min_distance_squared = std::numeric_limits<T>::max();
   
       // loop over each point in the current RHS space
       for (uint32_t rhs_p_idx = rhs_p_idx_begin; rhs_p_idx < rhs_p_idx_end; rhs_p_idx++) {
         // get the x and y coordinate of this RHS point
         auto const rhs_p_x = xs[rhs_p_idx];
         auto const rhs_p_y = ys[rhs_p_idx];
   
         // get distance between the LHS and RHS point
         auto const distance_squared = magnitude_squared(rhs_p_x - lhs_p_x, rhs_p_y - lhs_p_y);

         // remember only smallest distance from this LHS point to any RHS point.
         min_distance_squared = min(min_distance_squared, distance_squared);
       }
   
       // determine the output offset for this pair of spaces (LHS, RHS)
       auto output_idx = lhs_space_idx * num_spaces + rhs_space_idx;
   
       // use atomicMax to find the maximum of the minimum distance calculated for each space pair.
       if constexpr (kind == DISTANCE_KIND::HAUSDORFF) {
         atomicMax(results + output_idx, sqrt(min_distance_squared));
       } else {
         atomicMin(results + output_idx, sqrt(min_distance_squared));
       }
     }
   }
  
} // namespace detail
} // namespace cuspatial

