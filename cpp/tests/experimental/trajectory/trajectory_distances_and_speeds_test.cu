
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

#include "trajectory_test_utils.cuh"

#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/trajectory_distances_and_speeds.cuh>
#include <cuspatial/trajectory.hpp>
#include <cuspatial/vec_2d.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <gtest/gtest.h>

#include <limits>

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

    EXPECT_EQ(std::distance(distance_and_speed_begin, distance_and_speed_end),
              data.num_trajectories);

    auto [expected_distances, expected_speeds] = data.distance_and_speed();

    T max_expected_distance = thrust::reduce(expected_distances.begin(),
                                             expected_distances.end(),
                                             std::numeric_limits<T>::lowest(),
                                             thrust::maximum<T>{});

    T max_expected_speed =
      thrust::reduce(expected_speeds.begin(),
                     expected_speeds.end(),
                     std::numeric_limits<T>::lowest(),
                     [] __device__(T const& a, T const& b) { return max(abs(a), abs(b)); });

    // We expect the floating point error (in ulps) due to be proportional to the  number of
    // operations to compute the relevant quantity. For distance, this computation is
    // m_per_km * sqrt(dot(vec, vec)), where vec = (p1 - p0).
    // For speed, there is an additional division. There is also accumulated error in the reductions
    // and we find k_ulps == 10 reliably results in the expected computation matching the actual
    // computation for large trajectories with disparate positions and increasing timestamps.
    // This value and the magnitude of the values involed (e.g. max distance) are used to scale
    // the machine epsilon to come up with an absolute error tolerance.

    int k_ulps           = 10;
    T abs_error_distance = k_ulps * std::numeric_limits<T>::epsilon() * max_expected_distance;
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(distances, expected_distances, abs_error_distance);
    T abs_error_speed = k_ulps * std::numeric_limits<T>::epsilon() * max_expected_speed;
    CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(speeds, expected_speeds, abs_error_speed);
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

struct time_point_generator {
  using time_point = cuspatial::test::time_point;
  int init;

  time_point __device__ operator()(int const i)
  {
    return time_point{time_point::duration{i + init}};
  }
};

// Simple standalone test with hard-coded results
TYPED_TEST(TrajectoryDistancesAndSpeedsTest, ComputeDistanceAndSpeed3Simple)
{
  using T          = TypeParam;
  using time_point = cuspatial::test::time_point;

  std::int32_t num_trajectories = 3;

  auto offsets = rmm::device_vector<int32_t>{std::vector<std::int32_t>{0, 5, 9}};
  auto id =
    rmm::device_vector<int32_t>{std::vector<std::int32_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2}};
  auto points =
    rmm::device_vector<cuspatial::vec_2d<T>>(std::vector<cuspatial::vec_2d<T>>{{1.0, 0.0},
                                                                               {2.0, 1.0},
                                                                               {3.0, 2.0},
                                                                               {5.0, 3.0},
                                                                               {7.0, 1.0},
                                                                               {1.0, 3.0},
                                                                               {2.0, 5.0},
                                                                               {3.0, 6.0},
                                                                               {6.0, 5.0},
                                                                               {0.0, 4.0},
                                                                               {3.0, 7.0},
                                                                               {6.0, 4.0}});

  auto ts = rmm::device_vector<time_point>{12};
  thrust::tabulate(ts.begin(), ts.end(), time_point_generator{1});  // 1 through 12

  auto distances = rmm::device_vector<T>(num_trajectories);
  auto speeds    = rmm::device_vector<T>(num_trajectories);

  auto distance_and_speed_begin = thrust::make_zip_iterator(distances.begin(), speeds.begin());

  auto distance_and_speed_end = cuspatial::trajectory_distances_and_speeds(
    3, id.begin(), id.end(), points.begin(), ts.begin(), distance_and_speed_begin);

  ASSERT_EQ(std::distance(distance_and_speed_begin, distance_and_speed_end), num_trajectories);

  // expected distance and speed
  std::vector<T> speeds_expected({1973230.5567480423, 2270853.0666804211, 4242640.6871192846});
  std::vector<T> distances_expected({7892.9222269921693, 6812.5592000412635, 8485.2813742385697});

  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(distances, distances_expected);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(speeds, speeds_expected);
}
