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

#include <cuspatial/detail/iterator.hpp>
#include <cuspatial/experimental/trajectory_bounding_boxes.cuh>
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

#include <cstdint>

template <typename T>
struct DeriveTrajectoriesTest : public ::testing::Test {
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(DeriveTrajectoriesTest, TestTypes);

TYPED_TEST(DeriveTrajectoriesTest, OneMillionSmallTrajectories)
{
  using T   = TypeParam;
  auto data = cuspatial::test::trajectory_test_data<T>(1'000'000, 50);

  auto minima = rmm::device_vector<cuspatial::vec_2d<T>>(data.num_trajectories);
  auto maxima = rmm::device_vector<cuspatial::vec_2d<T>>(data.num_trajectories);

  auto [minima_end, maxima_end] = cuspatial::trajectory_bounding_boxes(data.ids_sorted.begin(),
                                                                       data.ids_sorted.end(),
                                                                       data.points_sorted.begin(),
                                                                       minima.begin(),
                                                                       maxima.begin());

  auto [expected_minima, expected_maxima] = data.extrema();

  EXPECT_EQ(std::distance(minima.begin(), minima_end), data.num_trajectories);
  EXPECT_EQ(std::distance(maxima.begin(), maxima_end), data.num_trajectories);

  EXPECT_EQ(minima, expected_minima);
  EXPECT_EQ(maxima, expected_maxima);
}

TYPED_TEST(DeriveTrajectoriesTest, OneHundredLargeTrajectories)
{
  using T   = TypeParam;
  auto data = cuspatial::test::trajectory_test_data<T>(100, 1'000'000);

  auto minima = rmm::device_vector<cuspatial::vec_2d<T>>(data.num_trajectories);
  auto maxima = rmm::device_vector<cuspatial::vec_2d<T>>(data.num_trajectories);

  auto [minima_end, maxima_end] = cuspatial::trajectory_bounding_boxes(data.ids_sorted.begin(),
                                                                       data.ids_sorted.end(),
                                                                       data.points_sorted.begin(),
                                                                       minima.begin(),
                                                                       maxima.begin());

  auto [expected_minima, expected_maxima] = data.extrema();

  EXPECT_EQ(std::distance(minima.begin(), minima_end), data.num_trajectories);
  EXPECT_EQ(std::distance(maxima.begin(), maxima_end), data.num_trajectories);

  EXPECT_EQ(minima, expected_minima);
  EXPECT_EQ(maxima, expected_maxima);
}

TYPED_TEST(DeriveTrajectoriesTest, OneVeryLargeTrajectory)
{
  using T   = TypeParam;
  auto data = cuspatial::test::trajectory_test_data<T>(1, 100'000'000);

  auto minima = rmm::device_vector<cuspatial::vec_2d<T>>(data.num_trajectories);
  auto maxima = rmm::device_vector<cuspatial::vec_2d<T>>(data.num_trajectories);

  auto [minima_end, maxima_end] = cuspatial::trajectory_bounding_boxes(data.ids_sorted.begin(),
                                                                       data.ids_sorted.end(),
                                                                       data.points_sorted.begin(),
                                                                       minima.begin(),
                                                                       maxima.begin());

  auto [expected_minima, expected_maxima] = data.extrema();

  EXPECT_EQ(std::distance(minima.begin(), minima_end), data.num_trajectories);
  EXPECT_EQ(std::distance(maxima.begin(), maxima_end), data.num_trajectories);

  EXPECT_EQ(minima, expected_minima);
  EXPECT_EQ(maxima, expected_maxima);
}
