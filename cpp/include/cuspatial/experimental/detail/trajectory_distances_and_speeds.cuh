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
#include <thrust/transform.h>

#include <cuda/std/chrono>

namespace cuspatial {

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

  // compute the per-point distance and duration using a 2-input thrust::transform
  // input: two copies of a zip iterator containing ids, points, and timestamps
  // second copy is offset by 1.  Only compute n - 1 outputs
  // output: zip iterator of durations and distances

  auto id_point_timestamp = thrust::make_zip_iterator(ids_first, points_first, timestamps_first);

  rmm::device_uvector<Rep> durations(num_points, stream);
  rmm::device_uvector<T> distances(num_points, stream);

  // initialize just the first elements since we will skip them
  durations.set_element_to_zero_async(0, stream);
  distances.set_element_to_zero_async(0, stream);

  auto duration_and_distance = thrust::make_zip_iterator(durations.begin(), distances.begin());

  thrust::transform(rmm::exec_policy(stream),
                    id_point_timestamp,
                    id_point_timestamp + num_points - 1,
                    id_point_timestamp + 1,
                    duration_and_distance + 1,
                    [] __device__(auto const& p0, auto const& p1) {
                      if (thrust::get<0>(p0) == thrust::get<0>(p1)) {  // ids are the same
                        Point pos0   = thrust::get<1>(p0);
                        Point pos1   = thrust::get<1>(p1);
                        Timestamp t0 = thrust::get<2>(p0);
                        Timestamp t1 = thrust::get<2>(p1);
                        Point vec    = pos1 - pos0;
                        // duration and distance
                        return thrust::make_tuple((t1 - t0).count(), sqrt(dot(vec, vec)));
                      }
                      return thrust::make_tuple(Rep{}, T{});
                    });

  auto duration_and_distance_tmp = thrust::make_zip_iterator(durations.begin(),
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
    typename cuda::std::ratio_divide<typename Timestamp::period, typename Sec::period>::type;

  // Reduce the intermediate durations and kilometer distances into meter
  // distances and speeds in meters/second
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        ids_first,
                        ids_last,
                        duration_and_distance_tmp,
                        thrust::discard_iterator(),
                        duration_distances_and_speed,
                        thrust::equal_to<Id>(),
                        [] __device__(auto a, auto b) {
                          auto time_d = Dur(thrust::get<0>(a)) + Dur(thrust::get<0>(b));
                          auto time_s = static_cast<T>(time_d.count()) *
                                        static_cast<T>(Period::num) / static_cast<T>(Period::den);
                          T dist_km   = thrust::get<1>(a) + thrust::get<1>(b);
                          T dist_m    = dist_km * T{1000.0};  // km to m
                          T speed_m_s = dist_m / time_s;      // m/ms to m/s
                          return thrust::make_tuple(time_d.count(), dist_km, dist_m, speed_m_s);
                        });

  // check for errors
  CUSPATIAL_CHECK_CUDA(stream.value());

  return std::next(distances_and_speeds_first, num_trajectories);
}

}  // namespace cuspatial
