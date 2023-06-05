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

#include "../trajectory/trajectory_test_utils.cuh"

#include <cuspatial_test/vector_equality.hpp>

#include <cuspatial/bounding_boxes.cuh>
#include <cuspatial/geometry/box.hpp>
#include <cuspatial/geometry/vec_2d.hpp>
#include <cuspatial/iterator_factory.cuh>

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
struct PointBoundingBoxesTest : public ::testing::Test {
  void run_test(int num_trajectories, int points_per_trajectory, T expansion_radius = T{})
  {
    auto data = cuspatial::test::trajectory_test_data<T>(num_trajectories, points_per_trajectory);

    auto bounding_boxes = rmm::device_vector<cuspatial::box<T>>(data.num_trajectories);

    auto boxes_end = cuspatial::point_bounding_boxes(data.ids_sorted.begin(),
                                                     data.ids_sorted.end(),
                                                     data.points_sorted.begin(),
                                                     bounding_boxes.begin());

    EXPECT_EQ(std::distance(bounding_boxes.begin(), boxes_end), data.num_trajectories);

    CUSPATIAL_EXPECT_VEC2D_PAIRS_EQUIVALENT(bounding_boxes, data.bounding_boxes());
  }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_CASE(PointBoundingBoxesTest, TestTypes);

TYPED_TEST(PointBoundingBoxesTest, TenThousandSmallTrajectories) { this->run_test(10'000, 50); }

TYPED_TEST(PointBoundingBoxesTest, OneHundredLargeTrajectories) { this->run_test(100, 10'000); }

TYPED_TEST(PointBoundingBoxesTest, OneVeryLargeTrajectory) { this->run_test(1, 1'000'000); }

TYPED_TEST(PointBoundingBoxesTest, TrajectoriesWithExpansion)
{
  this->run_test(10'000, 50, TypeParam{0.5});
}
