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
    cudf::column_view const& id, cudf::column_view const& x,
    cudf::column_view const& y, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
  using typename cudf::column;
  using typename cudf::table_view;
  using namespace cudf::experimental;

  groupby::groupby groupby{table_view{{id}}};

  std::vector<groupby::aggregation_request> reqs{};

  // count_ids aggregation
  reqs.push_back(groupby::aggregation_request{id});
  reqs[0].aggregations.push_back(make_count_aggregation());

  // min_timestamp aggregation
  reqs.push_back(groupby::aggregation_request{timestamp});
  reqs[1].aggregations.push_back(make_min_aggregation());

  // min_x aggregation
  reqs.push_back(groupby::aggregation_request{x});
  reqs[2].aggregations.push_back(make_min_aggregation());

  // min_y aggregation
  reqs.push_back(groupby::aggregation_request{y});
  reqs[3].aggregations.push_back(make_min_aggregation());

  // do the needful
  auto result = groupby.aggregate(reqs, mr);

  // extract the aggregation results
  auto ids = std::move(result.first->release()[0]);
  auto lengths = std::move(result.second[0].results[0]);
  auto min_ts = std::move(result.second[1].results[0]);
  auto min_x = std::move(result.second[2].results[0]);
  auto min_y = std::move(result.second[3].results[0]);

  // allocate_like to ensure the lengths null_mask is
  // retained. Offset sums are computed after the sort.
  auto offsets = cudf::experimental::allocate_like(
      *lengths, mask_allocation_policy::RETAIN, mr);

  // sort by id, timestamp, x, y
  auto output =
      sort_by_key(table_view{{*ids, *lengths, *offsets}},
                  table_view{{*ids, *min_ts, *min_x, *min_y}}, {}, {}, mr);

  // cumulative sum to fill offsets
  auto policy = rmm::exec_policy(stream);
  auto lens = output->get_column(1).view();
  auto offs = output->get_column(2).mutable_view();
  thrust::inclusive_scan(policy->on(stream), lens.begin<int32_t>(),
                         lens.end<int32_t>(), offs.begin<int32_t>());

  return output;
}
}  // namespace detail

std::unique_ptr<cudf::experimental::table> derive_trajectories(
    cudf::column_view const& id, cudf::column_view const& x,
    cudf::column_view const& y, cudf::column_view const& timestamp,
    rmm::mr::device_memory_resource* mr) {
  CUDF_EXPECTS(x.size() == y.size() && x.size() == id.size() &&
                   x.size() == timestamp.size(),
               "Data size mismatch");
  CUDF_EXPECTS(id.type().id() == cudf::INT32, "Invalid trajectory ID datatype");
  CUDF_EXPECTS(cudf::is_timestamp(timestamp.type()),
               "Invalid timestamp datatype");

  return detail::derive_trajectories(id, x, y, timestamp, mr, 0);
}
}  // namespace experimental
}  // namespace cuspatial
