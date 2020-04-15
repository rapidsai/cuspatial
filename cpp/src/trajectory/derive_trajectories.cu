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
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cuspatial/detail/trajectory.hpp>
#include <memory>
#include <vector>

namespace cuspatial {
namespace experimental {
namespace detail {

std::pair<std::unique_ptr<cudf::experimental::table>,
          std::unique_ptr<cudf::column>>
derive_trajectories(cudf::column_view const& x, cudf::column_view const& y,
                    cudf::column_view const& object_id,
                    cudf::column_view const& timestamp,
                    rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  auto sorted = cudf::experimental::detail::sort_by_key(
      cudf::table_view{{object_id, timestamp, x, y}},
      cudf::table_view{{object_id, timestamp}}, {}, {}, mr, stream);

  auto policy = rmm::exec_policy(stream);
  auto sorted_id = sorted->get_column(0).view();
  rmm::device_vector<int32_t> lengths(object_id.size());
  auto grouped = thrust::reduce_by_key(
      policy->on(stream), sorted_id.begin<int32_t>(), sorted_id.end<int32_t>(),
      thrust::make_constant_iterator(1), thrust::make_discard_iterator(),
      lengths.begin());

  auto offsets = cudf::make_numeric_column(
      cudf::data_type{cudf::INT32},
      thrust::distance(lengths.begin(), grouped.second),
      cudf::mask_state::UNALLOCATED, stream, mr);

  thrust::inclusive_scan(policy->on(stream), lengths.begin(), lengths.end(),
                         offsets->mutable_view().begin<int32_t>());

  return std::make_pair(std::move(sorted), std::move(offsets));
}
}  // namespace detail

std::pair<std::unique_ptr<cudf::experimental::table>,
          std::unique_ptr<cudf::column>>
derive_trajectories(cudf::column_view const& x, cudf::column_view const& y,
                    cudf::column_view const& object_id,
                    cudf::column_view const& timestamp,
                    rmm::mr::device_memory_resource* mr) {
  CUSPATIAL_EXPECTS(!(x.is_empty() || y.is_empty() || object_id.is_empty() ||
                      timestamp.is_empty()),
                    "Insufficient trajectory data");
  CUSPATIAL_EXPECTS(x.size() == y.size() && x.size() == object_id.size() &&
                        x.size() == timestamp.size(),
                    "Data size mismatch");
  CUSPATIAL_EXPECTS(cudf::is_timestamp(timestamp.type()),
                    "Invalid object_id datatype");
  CUSPATIAL_EXPECTS(object_id.type().id() == cudf::INT32,
                    "Invalid object_id datatype");
  CUSPATIAL_EXPECTS(!(x.has_nulls() || y.has_nulls() || object_id.has_nulls() ||
                      timestamp.has_nulls()),
                    "NULL support unimplemented");
  return detail::derive_trajectories(x, y, object_id, timestamp, mr, 0);
}
}  // namespace experimental
}  // namespace cuspatial
