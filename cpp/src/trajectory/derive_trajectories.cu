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

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <memory>
#include <vector>

#include "trajectory.hpp"

namespace cuspatial {
namespace experimental {
namespace detail {

std::unique_ptr<cudf::experimental::table> derive_trajectories(
    cudf::column_view const& id, rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {
  using typename cudf::column;
  using typename cudf::table_view;
  using namespace cudf::experimental;

  std::vector<groupby::aggregation_request> reqs{};
  reqs.reserve(2);

  // append count ids aggregation
  reqs.push_back(groupby::aggregation_request{id});
  reqs[0].aggregations.push_back(make_count_aggregation());

  // append nth_element aggregation to force method=sort
  reqs.push_back(groupby::aggregation_request{id});
  reqs[1].aggregations.push_back(make_nth_element_aggregation(0));

  // do the needful
  auto result = groupby::groupby{table_view{{id}}}.aggregate(reqs, mr);

  // extract the aggregation results
  std::vector<std::unique_ptr<cudf::column>> cols{};
  cols.reserve(3);
  // Append ids output column
  cols.push_back(std::move(result.first->release()[0]));
  // Append lengths output column
  cols.push_back(std::move(result.second[0].results[0]));
  // Append offsets output column. Use `allocate_like` to
  // ensure the `lengths` null_mask is retained if exists
  cols.push_back(cudf::experimental::allocate_like(
      *cols.at(1), mask_allocation_policy::RETAIN, mr));

  // cumulative sum to fill offsets
  auto policy = rmm::exec_policy(stream);
  thrust::exclusive_scan(policy->on(stream),
                         cols.at(1)->view().begin<int32_t>(),
                         cols.at(1)->view().end<int32_t>(),
                         cols.at(2)->mutable_view().begin<int32_t>());

  return std::make_unique<table>(std::move(cols));
}
}  // namespace detail

std::unique_ptr<cudf::experimental::table> derive_trajectories(
    cudf::column_view const& id, rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(id.type().id() == cudf::INT32, "Invalid trajectory ID datatype");
  return detail::derive_trajectories(id, mr, 0);
}
}  // namespace experimental
}  // namespace cuspatial
