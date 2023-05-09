/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "cuspatial_test/vector_equality.hpp"
#include "trajectory_test_utils.cuh"

#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>
#include <cuspatial/trajectory.cuh>

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
#include <iostream>

namespace std {

// Required by gtest EXPECT_EQ test suite to compile.
// Since `time_point` is an alias on
// std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>,
// according to ADL rules for templates, only the inner most enclosing namespaces,
// and associated namespaces of the template arguments are added to search. In this
// case, only `std` namespace is searched.
//
// [1]: https://en.cppreference.com/w/cpp/language/adl
std::ostream& operator<<(std::ostream& os, cuspatial::test::time_point const& tp)
{
  // Output the time point in the desired format
  os << tp.time_since_epoch().count() << "ms";
  return os;
}

}  // namespace std

template <typename T>
struct DeriveTrajectoriesTest : public ::testing::Test {};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(DeriveTrajectoriesTest, TestTypes);

TYPED_TEST(DeriveTrajectoriesTest, TenThousandSmallTrajectories)
{
  auto data = cuspatial::test::trajectory_test_data<TypeParam>(10'000, 50);

  auto traj_ids    = rmm::device_vector<std::int32_t>(data.ids.size());
  auto traj_points = rmm::device_vector<cuspatial::vec_2d<TypeParam>>(data.points.size());
  auto traj_times  = rmm::device_vector<cuspatial::test::time_point>(data.times.size());

  auto traj_offsets = cuspatial::derive_trajectories(data.ids.begin(),
                                                     data.ids.end(),
                                                     data.points.begin(),
                                                     data.times.begin(),
                                                     traj_ids.begin(),
                                                     traj_points.begin(),
                                                     traj_times.begin());

  EXPECT_EQ(traj_ids, data.ids_sorted);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(traj_points, data.points_sorted);
  EXPECT_EQ(traj_times, data.times_sorted);
}

TYPED_TEST(DeriveTrajectoriesTest, OneHundredLargeTrajectories)
{
  auto data = cuspatial::test::trajectory_test_data<TypeParam>(100, 10'000);

  auto traj_ids    = rmm::device_vector<std::int32_t>(data.ids.size());
  auto traj_points = rmm::device_vector<cuspatial::vec_2d<TypeParam>>(data.points.size());
  auto traj_times  = rmm::device_vector<cuspatial::test::time_point>(data.times.size());

  auto traj_offsets = cuspatial::derive_trajectories(data.ids.begin(),
                                                     data.ids.end(),
                                                     data.points.begin(),
                                                     data.times.begin(),
                                                     traj_ids.begin(),
                                                     traj_points.begin(),
                                                     traj_times.begin());

  EXPECT_EQ(traj_ids, data.ids_sorted);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(traj_points, data.points_sorted);
  EXPECT_EQ(traj_times, data.times_sorted);
}

TYPED_TEST(DeriveTrajectoriesTest, OneVeryLargeTrajectory)
{
  auto data = cuspatial::test::trajectory_test_data<TypeParam>(1, 1'000'000);

  auto traj_ids    = rmm::device_vector<std::int32_t>(data.ids.size());
  auto traj_points = rmm::device_vector<cuspatial::vec_2d<TypeParam>>(data.points.size());
  auto traj_times  = rmm::device_vector<cuspatial::test::time_point>(data.times.size());

  auto traj_offsets = cuspatial::derive_trajectories(data.ids.begin(),
                                                     data.ids.end(),
                                                     data.points.begin(),
                                                     data.times.begin(),
                                                     traj_ids.begin(),
                                                     traj_points.begin(),
                                                     traj_times.begin());

  EXPECT_EQ(traj_ids, data.ids_sorted);
  CUSPATIAL_EXPECT_VECTORS_EQUIVALENT(traj_points, data.points_sorted);
  EXPECT_EQ(traj_times, data.times_sorted);
}
