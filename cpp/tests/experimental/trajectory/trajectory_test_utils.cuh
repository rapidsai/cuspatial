
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

#include "thrust/iterator/discard_iterator.h"
#include <cuspatial/vec_2d.hpp>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <chrono>
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
      using cuspatial::min;
      return {min(min(p1, p2), p3), max(max(p1, p2), p4)};
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
};

}  // namespace test
}  // namespace cuspatial
