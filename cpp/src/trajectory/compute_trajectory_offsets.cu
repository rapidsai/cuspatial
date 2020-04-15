/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cudf/column/column_factories.hpp>
#include <cuspatial/detail/trajectory.hpp>
#include <memory>
#include <vector>

namespace cuspatial {
namespace experimental {
namespace detail {

std::unique_ptr<cudf::column> compute_trajectory_offsets(
    cudf::column_view const& object_id, rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
  auto policy = rmm::exec_policy(stream);
  rmm::device_vector<int32_t> lengths(object_id.size());
  auto grouped = thrust::reduce_by_key(
      policy->on(stream), object_id.begin<int32_t>(), object_id.end<int32_t>(),
      thrust::make_constant_iterator(1), thrust::make_discard_iterator(),
      lengths.begin());

  auto offsets = cudf::make_numeric_column(
      cudf::data_type{cudf::INT32},
      thrust::distance(lengths.begin(), grouped.second),
      cudf::mask_state::UNALLOCATED, stream, mr);

  thrust::inclusive_scan(policy->on(stream), lengths.begin(), lengths.end(),
                         offsets->mutable_view().begin<int32_t>());

  return offsets;
}
}  // namespace detail

std::unique_ptr<cudf::column> compute_trajectory_offsets(
    cudf::column_view const& object_id, rmm::mr::device_memory_resource* mr) {
  CUSPATIAL_EXPECTS(object_id.type().id() == cudf::INT32,
                    "Invalid object_id datatype");
  return detail::compute_trajectory_offsets(object_id, mr, 0);
}
}  // namespace experimental
}  // namespace cuspatial
