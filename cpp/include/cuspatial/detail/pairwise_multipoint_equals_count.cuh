/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuspatial/cuda_utils.hpp>
#include <cuspatial/detail/utility/zero_data.cuh>
#include <cuspatial/error.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/range/multipoint_range.cuh>
#include <cuspatial/range/range.cuh>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <ranger/ranger.hpp>

#include <iterator>
#include <type_traits>

namespace cuspatial {

namespace detail {

template <class MultiPointRangeA, class MultiPointRangeB, class OutputIt>
CUSPATIAL_KERNEL void pairwise_multipoint_equals_count_kernel(MultiPointRangeA lhs,
                                                              MultiPointRangeB rhs,
                                                              OutputIt output)
{
  using T = typename MultiPointRangeA::point_t::value_type;

  for (auto idx : ranger::grid_stride_range(lhs.num_points())) {
    auto geometry_id    = lhs.geometry_idx_from_point_idx(idx);
    vec_2d<T> lhs_point = lhs.point_begin()[idx];
    auto rhs_multipoint = rhs[geometry_id];

    atomicAdd(
      &output[geometry_id],
      thrust::binary_search(thrust::seq, rhs_multipoint.begin(), rhs_multipoint.end(), lhs_point));
  }
}

}  // namespace detail

template <class MultiPointRangeA, class MultiPointRangeB, class OutputIt>
OutputIt pairwise_multipoint_equals_count(MultiPointRangeA lhs,
                                          MultiPointRangeB rhs,
                                          OutputIt output,
                                          rmm::cuda_stream_view stream)
{
  using T       = typename MultiPointRangeA::point_t::value_type;
  using index_t = typename MultiPointRangeB::index_t;

  static_assert(is_same_floating_point<T, typename MultiPointRangeB::point_t::value_type>(),
                "Origin and input must have the same base floating point type.");

  CUSPATIAL_EXPECTS(lhs.size() == rhs.size(), "lhs and rhs inputs should have the same size.");

  if (lhs.size() == 0) return output;

  // Create a sorted copy of the rhs points.
  auto key_it = make_geometry_id_iterator<index_t>(rhs.offsets_begin(), rhs.offsets_end());

  rmm::device_uvector<index_t> rhs_keys(rhs.num_points(), stream);
  rmm::device_uvector<vec_2d<T>> rhs_point_sorted(rhs.num_points(), stream);

  thrust::copy(rmm::exec_policy(stream), key_it, key_it + rhs.num_points(), rhs_keys.begin());
  thrust::copy(
    rmm::exec_policy(stream), rhs.point_begin(), rhs.point_end(), rhs_point_sorted.begin());

  auto rhs_with_keys =
    thrust::make_zip_iterator(thrust::make_tuple(rhs_keys.begin(), rhs_point_sorted.begin()));

  thrust::sort(rmm::exec_policy(stream), rhs_with_keys, rhs_with_keys + rhs.num_points());

  auto rhs_sorted = multipoint_range{
    rhs.offsets_begin(), rhs.offsets_end(), rhs_point_sorted.begin(), rhs_point_sorted.end()};

  detail::zero_data_async(output, output + lhs.size(), stream);

  if (lhs.num_points() > 0) {
    auto [tpb, n_blocks] = grid_1d(lhs.num_points());
    detail::pairwise_multipoint_equals_count_kernel<<<n_blocks, tpb, 0, stream.value()>>>(
      lhs, rhs_sorted, output);

    CUSPATIAL_CHECK_CUDA(stream.value());
  }
  return output + lhs.size();
}

}  // namespace cuspatial
