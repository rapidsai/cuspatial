/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required point_b_y applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuspatial/error.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>

#include <memory>
#include <type_traits>

namespace {

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
 * @param num_points number of total points in the input (sum of points from all spaces)
 * @param xs x coordinates
 * @param ys y coordinates
 * @param num_spaces number of spaces in the input
 * @param space_offsets starting position of each first point in each space
 * @param results directed Hausdorff distances computed by kernel
 * @return
 */
template <typename T>
__global__ void kernel_hausdorff(size_type num_points,
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
    atomicMax(results + output_idx, sqrt(min_distance_squared));
  }
}

struct hausdorff_functor {
  template <typename T, typename... Args>
  std::enable_if_t<not std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Non-floating point operation is not supported");
  }

  template <typename T>
  std::enable_if_t<std::is_floating_point<T>::value, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& xs,
    cudf::column_view const& ys,
    cudf::column_view const& space_offsets,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto const num_points = static_cast<uint32_t>(xs.size());
    auto const num_spaces = static_cast<uint32_t>(space_offsets.size());

    CUSPATIAL_EXPECTS(num_spaces < (1 << 15), "Total number of spaces must be less than 2^16");

    auto const num_results = num_spaces * num_spaces;

    auto tid    = cudf::type_to_id<T>();
    auto result = cudf::make_fixed_width_column(
      cudf::data_type{tid}, num_results, cudf::mask_state::UNALLOCATED, stream, mr);

    if (result->size() == 0) { return result; }

    auto const result_view = result->mutable_view();

    // due to hausdorff kernel using `atomicMax` for output, the output must be initialized to <= 0
    // here the output is being initialized to -1, which should always be overwritten. If -1 is
    // found in the output, there is a bug where the output is not being written to in the hausdorff
    // kernel.
    thrust::fill(rmm::exec_policy(stream), result_view.begin<T>(), result_view.end<T>(), -1);

    auto const threads_per_block = 64;
    auto const num_tiles         = (num_points + threads_per_block - 1) / threads_per_block;

    auto kernel = kernel_hausdorff<T>;

    kernel<<<num_tiles, threads_per_block, 0, stream.value()>>>(
      num_points,
      xs.data<T>(),
      ys.data<T>(),
      num_spaces,
      space_offsets.begin<cudf::size_type>(),
      result_view.begin<T>());

    CUDA_TRY(cudaGetLastError());

    return result;
  }
};

}  // namespace

namespace cuspatial {

std::unique_ptr<cudf::column> directed_hausdorff_distance(cudf::column_view const& xs,
                                                          cudf::column_view const& ys,
                                                          cudf::column_view const& space_offsets,
                                                          rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(xs.type() == ys.type(), "Inputs `xs` and `ys` must have same type.");
  CUSPATIAL_EXPECTS(xs.size() == ys.size(), "Inputs `xs` and `ys` must have same length.");

  CUSPATIAL_EXPECTS(not xs.has_nulls() and not ys.has_nulls() and not space_offsets.has_nulls(),
                    "Inputs must not have nulls.");

  CUSPATIAL_EXPECTS(xs.size() >= space_offsets.size(),
                    "At least one point is required for each space");

  return cudf::type_dispatcher(
    xs.type(), hausdorff_functor(), xs, ys, space_offsets, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
