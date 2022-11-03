
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
    // operations to compute the relevant quantity. For distance, this is computation is
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
