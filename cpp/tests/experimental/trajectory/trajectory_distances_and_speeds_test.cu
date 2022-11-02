
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

#include "tests/utility/vector_equality.hpp"
#include "trajectory_test_utils.cuh"

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/trajectory_distances_and_speeds.cuh>
#include <cuspatial/trajectory.hpp>
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

template <typename T>
struct TrajectoryDistancesAndSpeedsTest : public ::testing::Test {
  void run_test(int num_trajectories, int points_per_trajectory)
  {
    auto data = cuspatial::test::trajectory_test_data<T>(num_trajectories, points_per_trajectory);

    auto distances = rmm::device_vector<T>(data.num_trajectories);
    auto speeds    = rmm::device_vector<T>(data.num_trajectories);

    auto distance_and_speed_begin = thrust::make_zip_iterator(distances.begin(), speeds.begin());

    auto distance_and_speed_end =
      cuspatial::trajectory_distances_and_speeds(data.num_trajectories,
                                                 data.ids_sorted.begin(),
                                                 data.ids_sorted.end(),
                                                 data.points_sorted.begin(),
                                                 data.times_sorted.begin(),
                                                 distance_and_speed_begin);

    auto [expected_distances, expected_speeds] = data.distance_and_speed();

    EXPECT_EQ(std::distance(distance_and_speed_begin, distance_and_speed_end),
              data.num_trajectories);

    cuspatial::test::expect_vector_equivalent(distances, expected_distances);
    cuspatial::test::expect_vector_equivalent(speeds, expected_speeds);
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(TrajectoryDistancesAndSpeedsTest, TestTypes);

TYPED_TEST(TrajectoryDistancesAndSpeedsTest, OneMillionSmallTrajectories)
{
  this->run_test(1'000'000, 50);
}

TYPED_TEST(TrajectoryDistancesAndSpeedsTest, OneHundredLargeTrajectories)
{
  this->run_test(100, 1'000'000);
}

TYPED_TEST(TrajectoryDistancesAndSpeedsTest, OneVeryLargeTrajectory)
{
  this->run_test(1, 100'000'000);
}
