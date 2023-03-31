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
#include <cuspatial/experimental/ranges/range.cuh>
#include <cuspatial/traits.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <iterator>
#include <type_traits>

namespace cuspatial {

template <class MultiPointRefA, class MultiPointRefB, class OutputIt>
OutputIt allpairs_multipoint_equals_count(MultiPointRefA const& lhs,
                                          MultiPointRefB const& rhs,
                                          OutputIt output,
                                          rmm::cuda_stream_view stream)
{
  using T = typename MultiPointRefA::point_t::value_type;

  static_assert(is_same_floating_point<T, typename MultiPointRefB::point_t::value_type>(),
                "Origin and input must have the same base floating point type.");

  if (lhs.size() == 0) return output;

  if (rhs.size() == 0) {
    detail::zero_data_async(output, output + lhs.size(), stream);
    return output + lhs.size();
  }

  rmm::device_uvector<vec_2d<T>> rhs_sorted(rhs.size(), stream);
  thrust::copy(rmm::exec_policy(stream), rhs.begin(), rhs.end(), rhs_sorted.begin());
  thrust::sort(rmm::exec_policy(stream), rhs_sorted.begin(), rhs_sorted.end());

  return thrust::transform(
    rmm::exec_policy(stream),
    lhs.begin(),
    lhs.end(),
    output,
    [rhs_sorted_range = range(rhs_sorted.begin(), rhs_sorted.end())] __device__(auto lhs_point) {
      auto [lower_it, upper_it] = thrust::equal_range(
        thrust::seq, rhs_sorted_range.cbegin(), rhs_sorted_range.cend(), lhs_point);
      return thrust::distance(lower_it, upper_it);
    });
}

}  // namespace cuspatial