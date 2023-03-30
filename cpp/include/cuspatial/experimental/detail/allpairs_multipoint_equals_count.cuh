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

#include <cuspatial/constants.hpp>
#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {

namespace detail {

template <class MultiPointRefA, class MultiPointRefB, class OutputIt>
void __global__ allpairs_point_equals_count_kernel(MultiPointRefA lhs,
                                                   MultiPointRefB rhs,
                                                   OutputIt output)
{
  using T = typename MultiPointRefA::point_t::value_type;

  static_assert(is_same_floating_point<T, typename MultiPointRefB::point_t::value_type>(),
                "Origin and input must have the same base floating point type.");

  for (auto idx = threadIdx.x + blockIdx.x * blockDim.x; idx < lhs.size() * rhs.size();
       idx += gridDim.x * blockDim.x) {
    vec_2d<T> lhs_point = *(lhs.point_tile_begin() + idx);
    vec_2d<T> rhs_point = *(rhs.point_repeat_begin(lhs.size()) + idx);

    atomicInc(&output[idx % lhs.size()], lhs_point == rhs_point);
  }
}

}  // namespace detail

template <class MultiPointRefA, class MultiPointRefB, class OutputIt>
OutputIt allpairs_multipoint_equals_count(MultiPointRefA lhs,
                                          MultiPointRefB rhs,
                                          OutputIt output,
                                          rmm::cuda_stream_view stream)
{
  using T = typename MultiPointRefA::point_t::value_type;

  static_assert(is_same_floating_point<T, typename MultiPointRefB::point_t::value_type>(),
                "Origin and input must have the same base floating point type.");

  detail::zero_data_async(output, output + lhs.size(), stream);

  auto [threads_per_block, block_size] = grid_1d(lhs.size() * rhs.size());
  detail::allpairs_point_equals_count_kernel<<<block_size, threads_per_block, 0, stream.value()>>>(
    lhs, rhs, output);

  CUSPATIAL_CHECK_CUDA(stream.value());
  return output + lhs.size();
}

}  // namespace cuspatial
