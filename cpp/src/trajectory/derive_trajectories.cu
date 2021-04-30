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

#include <cuspatial/error.hpp>
#include <cuspatial/trajectory.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <memory>
#include <vector>

namespace cuspatial {
namespace detail {

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> derive_trajectories(
  cudf::column_view const& object_id,
  cudf::column_view const& x,
  cudf::column_view const& y,
  cudf::column_view const& timestamp,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto sorted = cudf::detail::sort_by_key(cudf::table_view{{object_id, x, y, timestamp}},
                                          cudf::table_view{{object_id, timestamp}},
                                          {},
                                          {},
                                          stream,
                                          mr);

  auto sorted_id = sorted->get_column(0).view();
  rmm::device_vector<int32_t> lengths(object_id.size());
  auto grouped = thrust::reduce_by_key(rmm::exec_policy(stream),
                                       sorted_id.begin<int32_t>(),
                                       sorted_id.end<int32_t>(),
                                       thrust::make_constant_iterator(1),
                                       thrust::make_discard_iterator(),
                                       lengths.begin());

  auto offsets = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           thrust::distance(lengths.begin(), grouped.second),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);

  thrust::exclusive_scan(rmm::exec_policy(stream),
                         lengths.begin(),
                         lengths.end(),
                         offsets->mutable_view().begin<int32_t>());

  return std::make_pair(std::move(sorted), std::move(offsets));
}
}  // namespace detail

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> derive_trajectories(
  cudf::column_view const& object_id,
  cudf::column_view const& x,
  cudf::column_view const& y,
  cudf::column_view const& timestamp,
  rmm::mr::device_memory_resource* mr)
{
  CUSPATIAL_EXPECTS(
    x.size() == y.size() && x.size() == object_id.size() && x.size() == timestamp.size(),
    "Data size mismatch");
  CUSPATIAL_EXPECTS(object_id.type().id() == cudf::type_id::INT32, "Invalid object_id datatype");
  CUSPATIAL_EXPECTS(cudf::is_timestamp(timestamp.type()), "Invalid timestamp datatype");
  CUSPATIAL_EXPECTS(
    !(x.has_nulls() || y.has_nulls() || object_id.has_nulls() || timestamp.has_nulls()),
    "NULL support unimplemented");
  if (object_id.is_empty() || x.is_empty() || y.is_empty() || timestamp.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(4);
    cols.push_back(cudf::empty_like(object_id));
    cols.push_back(cudf::empty_like(x));
    cols.push_back(cudf::empty_like(y));
    cols.push_back(cudf::empty_like(timestamp));
    return std::make_pair(std::make_unique<cudf::table>(std::move(cols)),
                          cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32}));
  }
  return detail::derive_trajectories(object_id, x, y, timestamp, rmm::cuda_stream_default, mr);
}
}  // namespace cuspatial
