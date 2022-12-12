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
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/experimental/trajectory_distances_and_speeds.cuh>
#include <cuspatial/trajectory.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/tuple.h>

namespace cuspatial {

namespace {

template <typename T>
struct dispatch_timestamp {
  template <typename Timestamp>
  std::enable_if_t<cudf::is_timestamp<Timestamp>(), std::unique_ptr<cudf::table>> operator()(
    cudf::size_type num_trajectories,
    cudf::column_view const& object_id,
    cudf::column_view const& x,
    cudf::column_view const& y,
    cudf::column_view const& timestamp,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    // Construct output columns
    auto type = cudf::data_type{cudf::type_to_id<T>()};
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    // allocate distance output column
    cols.push_back(
      cudf::make_numeric_column(type, num_trajectories, cudf::mask_state::UNALLOCATED, stream, mr));
    // allocate speed output column
    cols.push_back(
      cudf::make_numeric_column(type, num_trajectories, cudf::mask_state::UNALLOCATED, stream, mr));

    auto points_begin = cuspatial::make_vec_2d_iterator(x.begin<T>(), y.begin<T>());

    trajectory_distances_and_speeds(
      num_trajectories,
      object_id.begin<cudf::size_type>(),
      object_id.end<cudf::size_type>(),
      points_begin,
      timestamp.begin<Timestamp>(),
      thrust::make_zip_iterator(cols.at(0)->mutable_view().begin<T>(),
                                cols.at(1)->mutable_view().begin<T>()),
      stream);

    // check for errors
    CUSPATIAL_CHECK_CUDA(stream.value());

    return std::make_unique<cudf::table>(std::move(cols));
  }

  template <typename Timestamp, typename... Args>
  std::enable_if_t<not cudf::is_timestamp<Timestamp>(), std::unique_ptr<cudf::table>> operator()(
    Args&&...)
  {
    CUSPATIAL_FAIL("Timestamp must be a timestamp type");
  }
};

struct dispatch_element {
  template <typename Element>
  std::enable_if_t<std::is_floating_point<Element>::value, std::unique_ptr<cudf::table>> operator()(
    cudf::size_type num_trajectories,
    cudf::column_view const& object_id,
    cudf::column_view const& x,
    cudf::column_view const& y,
    cudf::column_view const& timestamp,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr)
  {
    return cudf::type_dispatcher(timestamp.type(),
                                 dispatch_timestamp<Element>{},
                                 num_trajectories,
                                 object_id,
                                 x,
                                 y,
                                 timestamp,
                                 stream,
                                 mr);
  }

  template <typename Element, typename... Args>
  std::enable_if_t<not std::is_floating_point<Element>::value, std::unique_ptr<cudf::table>>
  operator()(Args&&...)
  {
    CUSPATIAL_FAIL("X and Y must be floating point types");
  }
};

}  // namespace

namespace detail {
std::unique_ptr<cudf::table> trajectory_distances_and_speeds(cudf::size_type num_trajectories,
                                                             cudf::column_view const& object_id,
                                                             cudf::column_view const& x,
                                                             cudf::column_view const& y,
                                                             cudf::column_view const& timestamp,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(
    x.type(), dispatch_element{}, num_trajectories, object_id, x, y, timestamp, stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::table> trajectory_distances_and_speeds(cudf::size_type num_trajectories,
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
    !(x.has_nulls() || y.has_nulls() || timestamp.has_nulls() || object_id.has_nulls()),
    "NULL support unimplemented");

  if (num_trajectories == 0 || x.is_empty() || y.is_empty() || object_id.is_empty() ||
      timestamp.is_empty()) {
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT64}));
    cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT64}));
    return std::make_unique<cudf::table>(std::move(cols));
  }

  return detail::trajectory_distances_and_speeds(
    num_trajectories, object_id, x, y, timestamp, rmm::cuda_stream_default, mr);
}

}  // namespace cuspatial
