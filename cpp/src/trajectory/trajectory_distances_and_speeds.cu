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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>

namespace cuspatial {

namespace {

template <typename T>
struct duplicate_first_element_func {
  cudf::column_device_view col;
  __device__ inline T operator()(cudf::size_type i)
  {
    return i > -1 ? col.element<T>(i) : col.element<T>(0);
  }
};

template <typename T>
auto duplicate_first_element_iterator(cudf::column_view const& col, rmm::cuda_stream_view stream)
{
  auto d_col = cudf::column_device_view::create(col, stream);
  return thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(-1),
                                         duplicate_first_element_func<T>{*d_col});
}

template <typename Element>
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
    std::vector<std::unique_ptr<cudf::column>> cols{};
    cols.reserve(2);
    // allocate distance output column
    cols.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                             num_trajectories,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr));
    // allocate speed output column
    cols.push_back(cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                             num_trajectories,
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr));

    using Rep     = typename Timestamp::rep;
    using Dur     = typename Timestamp::duration;
    using Seconds = typename cuda::std::chrono::seconds;

    rmm::device_vector<Rep> durations(x.size() + 1);
    rmm::device_vector<double> distances(x.size() + 1);

    auto timestamp_point_and_id = thrust::make_zip_iterator(
      thrust::make_tuple(duplicate_first_element_iterator<Rep>(timestamp, stream),  //
                         thrust::make_constant_iterator<double>(0.0),               //
                         duplicate_first_element_iterator<Element>(x, stream),      //
                         duplicate_first_element_iterator<Element>(y, stream),      //
                         duplicate_first_element_iterator<int32_t>(object_id, stream)));

    auto duration_and_distance_1 =
      thrust::make_zip_iterator(thrust::make_tuple(durations.begin(),  //
                                                   distances.begin(),  //
                                                   thrust::make_discard_iterator(),
                                                   thrust::make_discard_iterator(),
                                                   thrust::make_discard_iterator()));

    // Compute duration and distance difference between adjacent elements that
    // share the same object id
    thrust::adjacent_difference(rmm::exec_policy(stream),
                                timestamp_point_and_id,                     // first
                                timestamp_point_and_id + durations.size(),  // last
                                duration_and_distance_1,                    // result
                                [] __device__(auto next, auto curr) {       // binary_op
                                  int32_t id0 = thrust::get<4>(curr);
                                  int32_t id1 = thrust::get<4>(next);
                                  if (id0 == id1) {
                                    Timestamp t0 = Timestamp{Dur{thrust::get<0>(curr)}};
                                    Timestamp t1 = Timestamp{Dur{thrust::get<0>(next)}};
                                    auto x0      = static_cast<double>(thrust::get<2>(curr));
                                    auto x1      = static_cast<double>(thrust::get<2>(next));
                                    auto y0      = static_cast<double>(thrust::get<3>(curr));
                                    auto y1      = static_cast<double>(thrust::get<3>(next));
                                    return thrust::make_tuple((t1 - t0).count(),
                                                              hypot(x1 - x0, y1 - y0),  //
                                                              Element{},
                                                              Element{},
                                                              int32_t{});
                                  }
                                  return thrust::make_tuple(
                                    Rep{}, double{}, Element{}, Element{}, int32_t{});
                                });

    auto duration_and_distance_2 =
      thrust::make_zip_iterator(thrust::make_tuple(durations.begin(),
                                                   distances.begin(),
                                                   thrust::make_constant_iterator<double>(0),
                                                   thrust::make_constant_iterator<double>(0)));

    rmm::device_vector<Rep> durations_tmp(num_trajectories);
    rmm::device_vector<double> distances_tmp(num_trajectories);
    auto duration_distances_and_speed = thrust::make_zip_iterator(
      thrust::make_tuple(durations_tmp.begin(),                       // reduced duration
                         distances_tmp.begin(),                       // reduced distance
                         cols.at(0)->mutable_view().begin<double>(),  // distance
                         cols.at(1)->mutable_view().begin<double>())  // speed
    );

    using Period =
      typename cuda::std::ratio_divide<typename Timestamp::period, typename Seconds::period>::type;

    // Reduce the intermediate durations and kilometer distances into meter
    // distances and speeds in meters/second
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          object_id.begin<int32_t>(),       // keys_first
                          object_id.end<int32_t>(),         // keys_last
                          duration_and_distance_2 + 1,      // values_first
                          thrust::make_discard_iterator(),  // keys_output
                          duration_distances_and_speed,     // values_output
                          thrust::equal_to<int32_t>(),      // binary_pred
                          [] __device__(auto a, auto b) {   // binary_op
                            auto time_d = Dur(thrust::get<0>(a)) + Dur(thrust::get<0>(b));
                            auto time_s = static_cast<double>(time_d.count()) *
                                          static_cast<double>(Period::num) /
                                          static_cast<double>(Period::den);
                            double dist_km   = thrust::get<1>(a) + thrust::get<1>(b);
                            double dist_m    = dist_km * 1000.0;  // km to m
                            double speed_m_s = dist_m / time_s;   // m/ms to m/s
                            return thrust::make_tuple(time_d.count(), dist_km, dist_m, speed_m_s);
                          });

    // check for errors
    CHECK_CUDA(stream.value());

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
  CUSPATIAL_EXPECTS(object_id.type().id() == cudf::type_id::INT32, "Invalid object_id type");
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
