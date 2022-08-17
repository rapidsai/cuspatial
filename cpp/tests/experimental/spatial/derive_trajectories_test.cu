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
#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/derive_trajectories.cuh>
#include <cuspatial/vec_2d.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/scan.h>
#include <thrust/shuffle.h>

#include <gtest/gtest.h>

#include <cuda/std/chrono>

#include <cstdint>
#include <numeric>
#include <random>

template <typename T>
struct DeriveTrajectoriesTest : public ::testing::Test {
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(DeriveTrajectoriesTest, TestTypes);

template <typename T>
struct trajectory_test_data {
  using time_point = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>;

  rmm::device_vector<std::int32_t> offsets;

  rmm::device_vector<std::int32_t> ids;
  rmm::device_vector<time_point> times;
  rmm::device_vector<cuspatial::vec_2d<T>> points;

  rmm::device_vector<std::int32_t> ids_sorted;
  rmm::device_vector<time_point> times_sorted;
  rmm::device_vector<cuspatial::vec_2d<T>> points_sorted;

  trajectory_test_data(std::size_t num_trajectories, std::size_t max_trajectory_size)
  {
    make_data(num_trajectories, max_trajectory_size);
  }

  struct id_functor {
    int32_t* offsets_begin{};
    int32_t* offsets_end{};

    id_functor(int32_t* offsets_begin, int32_t* offsets_end)
      : offsets_begin(offsets_begin), offsets_end(offsets_end)
    {
    }

    __device__ int32_t operator()(int i)
    {
      return thrust::distance(
        offsets_begin,
        thrust::prev(thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, i)));
    }
  };

  struct timestamp_functor {
    int32_t* offsets_begin{};
    int32_t* offsets_end{};

    timestamp_functor(int32_t* offsets_begin, int32_t* offsets_end)
      : offsets_begin(offsets_begin), offsets_end(offsets_end)
    {
    }

    __device__ time_point operator()(int i, int id)
    {
      auto offset = thrust::prev(thrust::upper_bound(thrust::seq, offsets_begin, offsets_end, i));
      auto time_step = i - *offset;
      auto duration  = (id % 10) * std::chrono::milliseconds(1000) +
                      time_step * std::chrono::milliseconds(100) +
                      std::chrono::milliseconds(int(10 * cos(time_step)));
      return time_point{duration};
    }
  };

  struct point_functor {
    __device__ cuspatial::vec_2d<T> operator()(time_point const& time, int32_t id)
    {
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
    __device__ int32_t operator()(int i)
    {
      gen.discard(i);
      return size_rand(gen);
    }
  };

  void make_data(std::size_t num_trajectories, std::size_t max_trajectory_size)
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

    /*for (int i = 0; i < total_points; i++) {
      time_point tp              = times_sorted[i];
      cuspatial::vec_2d<T> point = points_sorted[i];
      std::cout << ids_sorted[i] << " " << tp.time_since_epoch().count() << " " << point.x << " "
                << point.y << std::endl;
    }

    std::cout << std::endl;

    for (int i = 0; i < total_points; i++) {
      time_point tp              = times[i];
      cuspatial::vec_2d<T> point = points[i];
      std::cout << ids[i] << " " << tp.time_since_epoch().count() << " " << point.x << " "
                << point.y << std::endl;
    }*/
  }
};

TYPED_TEST(DeriveTrajectoriesTest, OneMillionTrajectories)
{
  auto data = trajectory_test_data<TypeParam>(1'000'000, 50);

  auto traj_ids    = rmm::device_vector<int32_t>(data.ids.size());
  auto traj_points = rmm::device_vector<cuspatial::vec_2d<TypeParam>>(data.points.size());
  auto traj_times =
    rmm::device_vector<typename trajectory_test_data<TypeParam>::time_point>(data.times.size());

  auto traj_offsets = cuspatial::derive_trajectories(data.ids.begin(),
                                                     data.ids.end(),
                                                     data.points.begin(),
                                                     data.times.begin(),
                                                     traj_ids.begin(),
                                                     traj_points.begin(),
                                                     traj_times.begin());

  EXPECT_EQ(traj_ids, data.ids_sorted);
  EXPECT_EQ(traj_points, data.points_sorted);
  EXPECT_EQ(traj_times, data.times_sorted);
}
