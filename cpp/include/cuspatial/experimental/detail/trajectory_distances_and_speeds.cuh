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

#include <cuspatial/error.hpp>
#include <cuspatial/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <cuda/std/chrono>

namespace cuspatial {

namespace detail {

template <typename IndexType, typename Iter>
struct duplicate_first_element_func {
  Iter first;

  __device__ inline auto operator()(IndexType i) { return i > -1 ? first : first + i; }
};

template <typename IndexType, typename Iter>
auto make_duplicate_first_element_iterator(Iter first)
{
  return thrust::make_transform_iterator(thrust::make_counting_iterator<IndexType>(-1),
                                         duplicate_first_element_func<IndexType, Iter>{first});
}

}  // namespace detail

template <typename IdInputIt,
          typename PointInputIt,
          typename TimestampInputIt,
          typename OutputIt,
          typename IndexT>
OutputIt trajectory_distances_and_speeds(IndexT num_trajectories,
                                         IdInputIt ids_first,
                                         IdInputIt ids_last,
                                         PointInputIt points_first,
                                         TimestampInputIt timestamps_first,
                                         OutputIt distances_and_speeds_first,
                                         rmm::cuda_stream_view stream)
{
  using Id        = iterator_value_type<IdInputIt>;
  using Point     = iterator_value_type<PointInputIt>;
  using Timestamp = iterator_value_type<TimestampInputIt>;
  using T         = typename Point::value_type;

  using Rep = typename Timestamp::rep;
  using Dur = typename Timestamp::duration;
  using Sec = typename cuda::std::chrono::seconds;

  auto num_points = std::distance(ids_first, ids_last);

  rmm::device_uvector<Rep> durations(num_points + 1);
  rmm::device_uvector<T> distances(num_points + 1);

  auto timestamp_point_and_id = thrust::make_zip_iterator(
    detail::make_duplicate_first_element_iterator<Rep>(timestamps_first, stream),
    thrust::make_constant_iterator<T>(0.0),
    detail::make_duplicate_first_element_iterator<Point>(points_first, stream),
    detail::make_duplicate_first_element_iterator<Id>(ids_first, stream));

  auto duration_and_distance_1 = thrust::make_zip_iterator(durations.begin(),
                                                           distances.begin(),
                                                           thrust::make_discard_iterator(),
                                                           thrust::make_discard_iterator(),
                                                           thrust::make_discard_iterator());

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
                                  auto x0      = static_cast<T>(thrust::get<2>(curr));
                                  auto x1      = static_cast<T>(thrust::get<2>(next));
                                  auto y0      = static_cast<T>(thrust::get<3>(curr));
                                  auto y1      = static_cast<T>(thrust::get<3>(next));
                                  return thrust::make_tuple((t1 - t0).count(),
                                                            hypot(x1 - x0, y1 - y0),  //
                                                            Point{},
                                                            Id{});
                                }
                                return thrust::make_tuple(Rep{}, T{}, Point{}, Id{});
                              });

  auto duration_and_distance_2 = thrust::make_zip_iterator(durations.begin(),
                                                           distances.begin(),
                                                           thrust::make_constant_iterator<T>(0),
                                                           thrust::make_constant_iterator<T>(0));

  rmm::device_uvector<Rep> durations_tmp(num_trajectories, stream);
  rmm::device_uvector<T> distances_tmp(num_trajectories, stream);

  auto distances_begin = thrust::get<0>(distances_and_speeds_first.get_iterator_tuple());
  auto speeds_begin    = thrust::get<1>(distances_and_speeds_first.get_iterator_tuple());

  auto duration_distances_and_speed = thrust::make_zip_iterator(
    durations_tmp.begin(), distances_tmp.begin(), distances_begin, speeds_begin);

  using Period =
    typename cuda::std::ratio_divide<typename Timestamp::period, typename Seconds::period>::type;

  // Reduce the intermediate durations and kilometer distances into meter
  // distances and speeds in meters/second
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        ids_first,  // keys
                        ids_last,
                        duration_and_distance_2 + 1,      // values
                        thrust::make_discard_iterator(),  // keys_output
                        duration_distances_and_speed,     // values_output
                        thrust::equal_to<int32_t>(),      // binary_pred
                        [] __device__(auto a, auto b) {   // binary_op
                          auto time_d = Dur(thrust::get<0>(a)) + Dur(thrust::get<0>(b));
                          auto time_s = static_cast<double>(time_d.count()) *
                                        static_cast<double>(Period::num) /
                                        static_cast<double>(Period::den);
                          double dist_km   = thrust::get<1>(a) + thrust::get<1>(b);
                          double dist_m    = dist_km * T{1000.0};  // km to m
                          double speed_m_s = dist_m / time_s;      // m/ms to m/s
                          return thrust::make_tuple(time_d.count(), dist_km, dist_m, speed_m_s);
                        });

  // check for errors
  CUSPATIAL_CHECK_CUDA(stream.value());

  return std::next(distances_and_speeds_first, num_points);
}

}  // namespace cuspatial
