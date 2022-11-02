
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

#pragma once

#include "thrust/iterator/transform_iterator.h"
#include <cuspatial/experimental/iterator_factory.cuh>
#include <cuspatial/vec_2d.hpp>

#include <cuspatial_test/test_util.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cuda/std/chrono>

#include <cstdint>

namespace cuspatial {
namespace test {

using time_point = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

/* Test data generation for trajectory APIs. Generates num_trajectories trajectories of random
   size between 1 and max_trajectory_size samples. Creates both a reference (sorted) set of
   trajectory IDs, timestamps, and points, and a shuffled (unsorted) set of the same for test
   input. The sorted data can be input directly into `trajectory_distance_and_speed`, while
   the shuffled data can be input to `derive_trajectories`.

   The times are not random, but do vary somewhat between trajectories and trajectory lengths.
   Each trajectory has a distinct starting time point and trajectory timestamps increase
   monotonically at each sample. The interval between samples varies within a small range.

   Likewise, the positions are not random, but follow a sinusoid pattern based on the time stamps.
*/
template <typename T>
struct trajectory_test_data {
  std::size_t num_trajectories;

  rmm::device_vector<std::int32_t> offsets;

  rmm::device_vector<std::int32_t> ids;
  rmm::device_vector<time_point> times;
  rmm::device_vector<cuspatial::vec_2d<T>> points;

  rmm::device_vector<std::int32_t> ids_sorted;
  rmm::device_vector<time_point> times_sorted;
  rmm::device_vector<cuspatial::vec_2d<T>> points_sorted;

  trajectory_test_data(std::size_t num_trajectories, std::size_t max_trajectory_size)
    : num_trajectories(num_trajectories)
  {
    thrust::minstd_rand gen;
    thrust::uniform_int_distribution<std::int32_t> size_rand(1, max_trajectory_size);

    // random trajectory sizes
    rmm::device_vector<std::int32_t> sizes(num_trajectories);
    thrust::tabulate(
      rmm::exec_policy(), sizes.begin(), sizes.end(), size_rand_functor(gen, size_rand));

    // offset to each trajectory
    offsets.resize(num_trajectories);
    thrust::exclusive_scan(rmm::exec_policy(), sizes.begin(), sizes.end(), offsets.begin(), 0);
    auto total_points = sizes[num_trajectories - 1] + offsets[num_trajectories - 1];

    ids.resize(total_points);
    ids_sorted.resize(total_points);
    times.resize(total_points);
    times_sorted.resize(total_points);
    points.resize(total_points);
    points_sorted.resize(total_points);

    using namespace std::chrono_literals;

    thrust::tabulate(rmm::exec_policy(),
                     ids_sorted.begin(),
                     ids_sorted.end(),
                     id_functor(offsets.data().get(), offsets.data().get() + offsets.size()));

    thrust::transform(
      rmm::exec_policy(),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(total_points),
      ids_sorted.begin(),
      times_sorted.begin(),
      timestamp_functor(offsets.data().get(), offsets.data().get() + offsets.size()));

    thrust::transform(rmm::exec_policy(),
                      times_sorted.begin(),
                      times_sorted.end(),
                      ids_sorted.begin(),
                      points_sorted.begin(),
                      point_functor());

    // shuffle input data to create randomized order
    rmm::device_vector<std::int32_t> map(total_points);
    thrust::sequence(rmm::exec_policy(), map.begin(), map.end(), 0);
    thrust::shuffle(rmm::exec_policy(), map.begin(), map.end(), gen);

    thrust::gather(rmm::exec_policy(), map.begin(), map.end(), ids_sorted.begin(), ids.begin());
    thrust::gather(rmm::exec_policy(), map.begin(), map.end(), times_sorted.begin(), times.begin());
    thrust::gather(
      rmm::exec_policy(), map.begin(), map.end(), points_sorted.begin(), points.begin());
  }

  struct id_functor {
    std::int32_t* offsets_begin{};
    std::int32_t* offsets_end{};

    id_functor(std::int32_t* offsets_begin, std::int32_t* offsets_end)
      : offsets_begin(offsets_begin), offsets_end(offsets_end)
    {
    }

    __device__ std::int32_t operator()(int i)
    {
      // Find the index within the current trajectory
      return thrust::distance(
        offsets_begin,
        thrust::prev(thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, i)));
    }
  };

  struct timestamp_functor {
    std::int32_t* offsets_begin{};
    std::int32_t* offsets_end{};

    timestamp_functor(std::int32_t* offsets_begin, std::int32_t* offsets_end)
      : offsets_begin(offsets_begin), offsets_end(offsets_end)
    {
    }

    __device__ time_point operator()(int i, int id)
    {
      auto offset = thrust::prev(thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, i));
      auto time_step = i - *offset;
      // The arithmetic here just adds some variance to the time step but keeps it monotonically
      // increasing with `i`
      auto duration = (id % 10) * std::chrono::milliseconds(1000) +
                      time_step * std::chrono::milliseconds(100) +
                      std::chrono::milliseconds(int(10 * cos(time_step)));
      return time_point{duration};
    }
  };

  struct point_functor {
    __device__ cuspatial::vec_2d<T> operator()(time_point const& time, std::int32_t id)
    {
      // X is time in milliseconds, Y is cosine(time), offset by ID
      float duration = (time - time_point{std::chrono::milliseconds(0)}).count();
      return cuspatial::vec_2d<T>{duration / 1000, id + cos(duration)};
    }
  };

  struct size_rand_functor {
    thrust::minstd_rand gen;
    thrust::uniform_int_distribution<std::int32_t> size_rand;

    size_rand_functor(thrust::minstd_rand gen,
                      thrust::uniform_int_distribution<std::int32_t> size_rand)
      : gen(gen), size_rand(size_rand)
    {
    }
    __device__ std::int32_t operator()(int i)
    {
      gen.discard(i);
      return size_rand(gen);
    }
  };

  struct box_minmax {
    using point_tuple = thrust::tuple<cuspatial::vec_2d<T>, cuspatial::vec_2d<T>>;
    __host__ __device__ point_tuple operator()(point_tuple const& a, point_tuple const& b)
    {
      vec_2d<T> p1, p2, p3, p4;
      thrust::tie(p1, p2) = a;
      thrust::tie(p3, p4) = b;
      return {box_min(box_min(p1, p2), p3), box_max(box_max(p1, p2), p4)};
    }
  };

  auto extrema()
  {
    auto minima = rmm::device_vector<cuspatial::vec_2d<T>>(num_trajectories);
    auto maxima = rmm::device_vector<cuspatial::vec_2d<T>>(num_trajectories);

    auto point_tuples = thrust::make_zip_iterator(points_sorted.begin(), points_sorted.begin());

    thrust::reduce_by_key(ids_sorted.begin(),
                          ids_sorted.end(),
                          point_tuples,
                          thrust::discard_iterator{},
                          thrust::make_zip_iterator(minima.begin(), maxima.begin()),
                          thrust::equal_to<std::int32_t>(),
                          box_minmax{});

    return std::pair{minima, maxima};
  }

  struct duration_functor {
    using id_and_timestamp = thrust::tuple<std::int32_t, time_point>;

    __host__ __device__ time_point::rep operator()(id_and_timestamp const& p0,
                                                   id_and_timestamp const& p1)
    {
      auto const id0 = thrust::get<0>(p0);
      auto const id1 = thrust::get<0>(p1);
      auto const t0  = thrust::get<1>(p0);
      auto const t1  = thrust::get<1>(p1);

      // printf("%d: %d %s %d\n", threadIdx.x, id0, (id0 == id1 ? "==" : "!="), id1);
      if (id0 == id1) {
        /*printf("%ld - %ld = %ld\n",
               t1.time_since_epoch().count(),
               t0.time_since_epoch().count(),
               (t1 - t0).count());*/
        return (t1 - t0).count();
      }
      return 0;
    }
  };

  struct distance_functor {
    using id_and_position = thrust::tuple<std::int32_t, cuspatial::vec_2d<T>>;
    __host__ __device__ T operator()(id_and_position const& p0, id_and_position const& p1)
    {
      auto const id0 = thrust::get<0>(p0);
      auto const id1 = thrust::get<0>(p1);
      if (id0 == id1) {
        auto const pos0 = thrust::get<1>(p0);
        auto const pos1 = thrust::get<1>(p1);
        return hypot(pos1.x - pos0.x, pos1.y - pos0.y);
      }
      return 0;
    }
  };

  struct average_distance_speed_functor {
    using duration_distance = thrust::tuple<time_point::rep, T, T, T>;
    using Sec               = typename cuda::std::chrono::seconds;
    using Period =
      typename cuda::std::ratio_divide<typename time_point::period, typename Sec::period>::type;

    __host__ __device__ duration_distance operator()(duration_distance const& a,
                                                     duration_distance const& b)
    {
      auto time_d =
        time_point::duration(thrust::get<0>(a)) + time_point::duration(thrust::get<0>(b));
      auto time_s = static_cast<double>(time_d.count()) * static_cast<double>(Period::num) /
                    static_cast<double>(Period::den);
      double dist_km   = thrust::get<1>(a) + thrust::get<1>(b);
      double dist_m    = dist_km * T{1000.0};  // km to m
      double speed_m_s = dist_m / time_s;      // m/ms to m/s
      return {time_d.count(), dist_km, dist_m, speed_m_s};
    }
  };

  std::pair<rmm::device_vector<T>, rmm::device_vector<T>> distance_and_speed()
  {
    using Rep = typename time_point::rep;

    /*cuspatial::test::print_device(ids_sorted.begin(), ids_sorted.end());
    auto time_print_begin = thrust::make_transform_iterator(
      times_sorted.begin(), [] __device__(auto const& t) { return t.time_since_epoch().count(); });
    cuspatial::test::print_device(time_print_begin, time_print_begin + times_sorted.size());
    cuspatial::test::print_device(points_sorted.begin(), points_sorted.end());*/

    auto id_and_timestamp = thrust::make_zip_iterator(ids_sorted.begin(), times_sorted.begin());

    auto duration_per_step = rmm::device_vector<Rep>(points.size());

    thrust::transform(rmm::exec_policy(),
                      id_and_timestamp,
                      id_and_timestamp + points.size() - 1,
                      id_and_timestamp + 1,
                      duration_per_step.begin() + 1,
                      duration_functor{});

    // cuspatial::test::print_device(duration_per_step.begin(), duration_per_step.end());

    auto id_and_position = thrust::make_zip_iterator(ids_sorted.begin(), points_sorted.begin());

    auto distance_per_step = rmm::device_vector<T>{points.size()};

    thrust::transform(rmm::exec_policy(),
                      id_and_position,
                      id_and_position + points.size() - 1,
                      id_and_position + 1,
                      distance_per_step.begin() + 1,
                      distance_functor{});

    // cuspatial::test::print_device(distance_per_step.begin(), distance_per_step.end());

    rmm::device_vector<Rep> durations_tmp(offsets.size());
    rmm::device_vector<T> distances_tmp(offsets.size());

    rmm::device_vector<T> distances(offsets.size());
    rmm::device_vector<T> speeds(offsets.size());

    auto duration_distance_and_speed = thrust::make_zip_iterator(
      durations_tmp.begin(), distances_tmp.begin(), distances.begin(), speeds.begin());

    auto duration_and_distance_init =
      thrust::make_zip_iterator(duration_per_step.begin(),
                                distance_per_step.begin(),
                                thrust::make_constant_iterator<T>(0),
                                thrust::make_constant_iterator<T>(0));

    thrust::reduce_by_key(rmm::exec_policy(),
                          ids_sorted.begin(),
                          ids_sorted.end(),
                          duration_and_distance_init,
                          thrust::discard_iterator{},
                          duration_distance_and_speed,
                          thrust::equal_to<int32_t>(),        // binary_pred
                          average_distance_speed_functor{});  // binary_op

    CUSPATIAL_CHECK_CUDA(rmm::cuda_stream_default);

    return std::pair{distances, speeds};
  }
};

}  // namespace test
}  // namespace cuspatial
