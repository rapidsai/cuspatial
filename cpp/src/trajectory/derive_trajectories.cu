/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <cuspatial/experimental/derive_trajectories.cuh>
#include <cuspatial/experimental/iterator_factory.cuh>

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>
#include <vector>

namespace cuspatial {
namespace detail {

struct derive_trajectories_dispatch {
  template <
    typename T,
    typename Timestamp,
    std::enable_if_t<std::is_floating_point_v<T> and cudf::is_timestamp<Timestamp>()>* = nullptr>
  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> operator()(
    cudf::column_view const& object_id,
    cudf::column_view const& x,
    cudf::column_view const& y,
    cudf::column_view const& timestamp,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    auto cols = std::vector<std::unique_ptr<cudf::column>>{};
    cols.reserve(4);
    cols.push_back(cudf::allocate_like(object_id, cudf::mask_allocation_policy::NEVER, mr));
    cols.push_back(cudf::allocate_like(x, cudf::mask_allocation_policy::NEVER, mr));
    cols.push_back(cudf::allocate_like(y, cudf::mask_allocation_policy::NEVER, mr));
    cols.push_back(cudf::allocate_like(timestamp, cudf::mask_allocation_policy::NEVER, mr));

    auto points_begin     = thrust::make_zip_iterator(x.begin<T>(), y.begin<T>());
    auto points_out_begin = thrust::make_zip_iterator(cols[1]->mutable_view().begin<T>(),
                                                      cols[2]->mutable_view().begin<T>());

    auto offsets = derive_trajectories(object_id.begin<std::int32_t>(),
                                       object_id.end<std::int32_t>(),
                                       points_begin,
                                       timestamp.begin<Timestamp>(),
                                       cols[0]->mutable_view().begin<std::int32_t>(),
                                       points_out_begin,
                                       cols[3]->mutable_view().begin<Timestamp>(),
                                       stream,
                                       mr);

    auto result_table = std::make_unique<cudf::table>(std::move(cols));

    auto num_trajectories = offsets->size();

    auto offsets_column = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, num_trajectories, offsets->release());

    return {std::move(result_table), std::move(offsets_column)};
  }

  template <typename T,
            typename Timestamp,
            std::enable_if_t<not(std::is_floating_point_v<T> and
                                 cudf::is_timestamp<Timestamp>())>* = nullptr>
  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> operator()(...)
  {
    CUSPATIAL_FAIL("Unsupported data type");
  }
};

std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::column>> derive_trajectories(
  cudf::column_view const& object_id,
  cudf::column_view const& x,
  cudf::column_view const& y,
  cudf::column_view const& timestamp,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  return cudf::double_type_dispatcher(x.type(),
                                      timestamp.type(),
                                      derive_trajectories_dispatch{},
                                      object_id,
                                      x,
                                      y,
                                      timestamp,
                                      stream,
                                      mr);
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
  CUSPATIAL_EXPECTS(x.type().id() == y.type().id(), "Data type mismatch");
  CUSPATIAL_EXPECTS(object_id.type().id() == cudf::type_to_id<cudf::size_type>(),
                    "Invalid object_id type");
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
