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

#include <cuspatial/vec_2d.hpp>
#include <cuspatial/error.hpp>
#include <cuspatial/experimental/spatial_window.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <type_traits>

#include <gtest/gtest.h>

template <typename T>
struct SpatialWindowTest : public testing::Test {
};

using TestTypes = ::testing::Types<float, double>;

TYPED_TEST_CASE(SpatialWindowTest, TestTypes);

TYPED_TEST(SpatialWindowTest, Empty)
{
  using T      = TypeParam;
  using Vec    = cuspatial::vec_2d<T>;
  using VecVec = std::vector<Vec>;
  auto points  = rmm::device_vector<Vec>{};

  auto result_size = cuspatial::count_points_in_spatial_window(
    Vec{1.5, 1.5}, Vec{5.5, 5.5}, points.begin(), points.end());

  EXPECT_EQ(result_size, 0);

  auto expected_points = rmm::device_vector<Vec>{};
  auto result_points   = rmm::device_vector<Vec>{};

  cuspatial::points_in_spatial_window(
    Vec{1.5, 1.5}, Vec{5.5, 5.5}, points.begin(), points.end(), result_points.begin());

  EXPECT_EQ(expected_points, result_points);
}

TYPED_TEST(SpatialWindowTest, SimpleTest)
{
  using T      = TypeParam;
  using Vec    = cuspatial::vec_2d<T>;
  using VecVec = std::vector<Vec>;
  auto points  = rmm::device_vector<Vec>(VecVec({{1.0, 0.0},
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
                                                {6.0, 4.0}}));

  auto expected_points = rmm::device_vector<Vec>(VecVec({{3.0, 2.0}, {5.0, 3.0}, {2.0, 5.0}}));

  auto result_size = cuspatial::count_points_in_spatial_window(
    Vec{1.5, 1.5}, Vec{5.5, 5.5}, points.begin(), points.end());

  EXPECT_EQ(result_size, expected_points.size());

  auto result_points = rmm::device_vector<Vec>(result_size);
  cuspatial::points_in_spatial_window(
    Vec{1.5, 1.5}, Vec{5.5, 5.5}, points.begin(), points.end(), result_points.begin());

  EXPECT_EQ(expected_points, result_points);
}

// Test that windows with min/max reversed still work
TYPED_TEST(SpatialWindowTest, ReversedWindow)
{
  using T      = TypeParam;
  using Vec    = cuspatial::vec_2d<T>;
  using VecVec = std::vector<Vec>;
  auto points  = rmm::device_vector<Vec>(VecVec({{1.0, 0.0},
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
                                                {6.0, 4.0}}));

  auto expected_points = rmm::device_vector<Vec>(VecVec({{3.0, 2.0}, {5.0, 3.0}, {2.0, 5.0}}));

  auto result_size = cuspatial::count_points_in_spatial_window(
    Vec{5.5, 5.5}, Vec{1.5, 1.5}, points.begin(), points.end());

  EXPECT_EQ(result_size, expected_points.size());

  auto result_points = rmm::device_vector<Vec>(result_size);
  cuspatial::points_in_spatial_window(
    Vec{5.5, 5.5}, Vec{1.5, 1.5}, points.begin(), points.end(), result_points.begin());

  EXPECT_EQ(expected_points, result_points);
}
